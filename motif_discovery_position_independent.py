#!/usr/bin/env python3
"""
Self-Attention Based Motif Discovery - Position Independent Approach
====================================================================

Simplified implementation using ONLY the position-independent discovery approach
with consolidation enabled by default.

Implementation based on "Exploring Language Models for Motif Discovery in
Immunopeptidomics Datasets" by Eloise Milliken (2021).

Features:
- CSV data loading capability
- Variable-only masking (never masks structural A/C positions)
- Position-independent motif discovery
- Automatic motif consolidation (parent-child hierarchy)
- Length-stratified PSSM visualization
- Interactive visualizations (network, tree, dashboard)
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from transformers import (
    RobertaConfig, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    PreTrainedTokenizer
)
from datasets import Dataset
import torch
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import argparse
import sys
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import csv
import re
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyvis.network import Network
import warnings

# ============================================================================
# Hierarchy Reconstruction Functions (from webapp's data_pipeline.py)
# ============================================================================

TOKEN_PATTERN = re.compile(r"\[[0-9]+\]|[A-Z]")


def tokenize_motif(motif: str) -> List[str]:
    """Split motif strings into comparable tokens."""
    if not isinstance(motif, str):
        return []
    motif = motif.strip().upper()
    return TOKEN_PATTERN.findall(motif)


def normalized_motif(motif: str) -> str:
    """Return a compact token string for substring comparisons."""
    return "".join(tokenize_motif(motif))


def is_parent_of(parent: str, child: str) -> bool:
    """Heuristic parent/child relationship based on token containment."""
    if not parent or not child or parent == child:
        return False
    parent_norm = normalized_motif(parent)
    child_norm = normalized_motif(child)
    if not parent_norm or not child_norm:
        return False
    if len(parent_norm) >= len(child_norm):
        return False
    return parent_norm in child_norm


def build_hierarchy_from_motifs(motifs: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Determine parent-child relationships among motifs using substring matching.

    Args:
        motifs: List of motif pattern strings

    Returns:
        hierarchy: Dict mapping parent motifs to list of their children
        parent_map: Dict mapping child motifs to their most immediate parent
    """
    hierarchy: Dict[str, List[str]] = {}
    parent_map: Dict[str, str] = {}

    for i, parent in enumerate(motifs):
        children = []
        for j, child in enumerate(motifs):
            if i == j:
                continue
            if is_parent_of(parent, child):
                children.append(child)
                # Choose longest parent (most specific) if child has multiple parents
                current = parent_map.get(child)
                if current is None or len(parent) > len(current):
                    parent_map[child] = parent
        if children:
            hierarchy[parent] = sorted(set(children))

    return hierarchy, parent_map

# ============================================================================

class ProteinRobertaTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for protein sequences that's compatible with HuggingFace"""

    def __init__(self, **kwargs):
        # 20 amino acids + special tokens
        self.amino_acids = "ARNDCQEGHILKMFPSTWYV"

        # Create vocabulary
        vocab = {
            '<s>': 0,
            '</s>': 1,
            '<mask>': 2,
            '<pad>': 3,
            '<unk>': 4
        }
        for i, aa in enumerate(self.amino_acids):
            vocab[aa] = i + 5

        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}

        # Initialize parent class
        super().__init__(
            bos_token='<s>',
            eos_token='</s>',
            mask_token='<mask>',
            pad_token='<pad>',
            unk_token='<unk>',
            **kwargs
        )

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return self._vocab.copy()

    def _tokenize(self, text):
        """Tokenize protein sequence into individual amino acids"""
        return list(text.upper())

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab['<unk>'])

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, '<unk>')

    def convert_tokens_to_string(self, tokens):
        """Convert tokens back to sequence"""
        return ''.join([t for t in tokens if t not in ['<s>', '</s>', '<pad>', '<unk>']])

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Add special tokens around the sequence"""
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + bos + token_ids_1 + eos

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save vocabulary to directory - required by HuggingFace"""
        import os
        import json

        if filename_prefix is None:
            filename_prefix = ""

        vocab_file = os.path.join(save_directory, f"{filename_prefix}vocab.json")

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

class SelfAttentionMotifDiscovery:
    """
    Self-attention based motif discovery following Milliken et al. 2021
    """

    def __init__(self, num_models: int = 5,  max_length: int = 36):
        self.num_models = num_models
        self.max_length = 36 # Fallback default
        self.tokenizer = ProteinRobertaTokenizer()
        self.models = []
        self.motifs = []
        self.attention_threshold = None
        self.valid_amino_acids = set("ARNDCQEGHILKMFPSTWYV")
        self.mask_structural_positions = True  # Enable variable-only masking
        self.structural_positions_cache = {}  # Pre-calculated structural positions
        self.build_pssms_during_discovery = False  # Defer PSSM building until after consolidation

        # For comprehensive analysis
        self.last_attention_vectors = None
        self.last_sequences = None
        self.analysis_results = {}  # Store results from different approaches

    def load_sequences_from_csv(self, csv_file: str, sequence_column: str) -> List[str]:
        """Load protein sequences from a CSV file"""
        print(f"Loading sequences from {csv_file}, column: {sequence_column}")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        if sequence_column not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise KeyError(f"Column '{sequence_column}' not found. Available columns: {available_columns}")

        sequences = df[sequence_column].tolist()

        validated_sequences = []
        invalid_sequences = []
        non_conforming_sequences = []

        for i, seq in enumerate(sequences):
            if pd.isna(seq) or seq == "":
                raise ValueError(f"Missing or empty sequence at row {i+1}")

            seq = str(seq).upper().strip()

            # Check for invalid amino acids
            invalid_chars = set(seq) - self.valid_amino_acids
            if invalid_chars:
                invalid_sequences.append((i+1, seq, invalid_chars))
                continue

            # ADD NEW: Check for conforming structure (AC...C...CA with exactly 3 cysteines)
            if not (seq.startswith('AC') and seq.endswith('A') and seq.count('C') == 3):
                non_conforming_sequences.append((i+1, seq, "doesn't match AC...C...CA pattern"))
                continue

            # Check cysteine positions for valid loop structure
            c_positions = [j for j, aa in enumerate(seq) if aa == 'C']
            if len(c_positions) == 3 and c_positions[0] == 1: # First C at position 1
                if c_positions[1] > c_positions[0] + 1 and c_positions[2] > c_positions[1] + 1:
                    validated_sequences.append(seq)
                else:
                    non_conforming_sequences.append((i+1, seq, "invalid cysteine spacing"))
            else:
                non_conforming_sequences.append((i+1, seq, "invalid cysteine positioning"))

        # Handle invalid amino acids (existing error behavior)
        if invalid_sequences:
            error_msg = "Invalid amino acid characters found in sequences:\n"
            for row, seq, invalid_chars in invalid_sequences[:5]:
                error_msg += f" Row {row}: '{seq}' contains invalid characters: {invalid_chars}\n"
            if len(invalid_sequences) > 5:
                error_msg += f" ... and {len(invalid_sequences) - 5} more invalid sequences\n"
            error_msg += f"Valid amino acids are: {sorted(self.valid_amino_acids)}"
            raise ValueError(error_msg)

        # Report non-conforming sequences but continue (new behavior)
        if non_conforming_sequences:
            print(f"Warning: Filtered out {len(non_conforming_sequences)} non-conforming sequences:")
            for row, seq, reason in non_conforming_sequences[:5]:
                print(f" Row {row}: '{seq}' - {reason}")
            if len(non_conforming_sequences) > 5:
                print(f" ... and {len(non_conforming_sequences) - 5} more")

        if not validated_sequences:
            raise ValueError("No conforming sequences found. All sequences must follow AC...C...CA pattern with exactly 3 cysteines.")

        print(f"✅ Successfully loaded {len(validated_sequences)} conforming sequences")
        print(f"📊 Sequence length range: {min(len(s) for s in validated_sequences)} - {max(len(s) for s in validated_sequences)}")

        return validated_sequences

    def create_test_dataset(self) -> Tuple[List[str], List[str]]:
        """Create test dataset for validation"""
        sequences = []
        labels = []

        templates = [
            "ACXXFXCXXKXXXCA",
            "ACXXXFCXXXKXXCA",
            "ACXFXXCXXXKXXCA",
        ]

        amino_acids = "ARNDCQEGHILKMFPSTWYV"
        np.random.seed(42)

        template_names = ["W4+K8", "W3+K8", "W4+K7"]

        for template_idx, template in enumerate(templates):
            for _ in range(1000):
                seq = ""
                for char in template:
                    if char == 'X':
                        seq += np.random.choice(list(amino_acids))
                    else:
                        seq += char
                sequences.append(seq)
                labels.append(template_names[template_idx])

        for _ in range(5000):
            length = np.random.randint(8, 12)
            seq = ''.join(np.random.choice(list(amino_acids)) for _ in range(length))
            sequences.append(seq)
            labels.append("random")

        return sequences, labels

    def get_structural_mask_positions(self, sequence: str) -> List[int]:
        """Get positions to mask for structural elements: first position, last position, and all C"""
        mask_positions = []

        # Always mask first and last positions (framework)
        if len(sequence) > 0:
            mask_positions.append(0)  # First position
        if len(sequence) > 1:
            mask_positions.append(len(sequence) - 1)  # Last position

        # All cysteines
        for i, aa in enumerate(sequence):
            if aa == 'C':
                mask_positions.append(i)

        return mask_positions

    def create_model_config(self) -> RobertaConfig:
        """Create RoBERTa configuration for protein sequences"""
        config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=self.max_length,
            type_vocab_size=1,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return config

    def prepare_dataset(self, sequences: List[str]) -> Dataset:
        """Prepare dataset for training"""
        logging.info("📊 Preparing dataset and pre-calculating structural positions...")

        # Pre-calculate structural positions for all sequences (always do this for consistency)
        for i, seq in enumerate(sequences):
            structural_positions = self.get_structural_mask_positions(seq)
            self.structural_positions_cache[i] = set(structural_positions)

            # Log first few for debugging
            if i < 5:
                logging.info(f"Sequence {i}: '{seq}' -> structural positions: {structural_positions}")

        encoded_data = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )

        # Create dataset - only add sequence_index if we need it for structural masking
        dataset_dict = {
            'input_ids': encoded_data['input_ids'],
            'attention_mask': encoded_data['attention_mask']
        }

        if self.mask_structural_positions:
            dataset_dict['sequence_index'] = list(range(len(sequences)))  # Only add when needed
            logging.info("✅ Added sequence_index to dataset for structural masking")

        dataset = Dataset.from_dict(dataset_dict)

        logging.info(f"✅ Dataset prepared with {len(sequences)} sequences")
        return dataset

    def train_models(self, sequences: List[str], epochs: int = 20,
                    enable_validation: bool = False, validation_split: float = 0.2,
                    early_stopping_patience: int = 3, output_dir: Optional[Path] = None):
        """
        Train multiple RoBERTa models on the sequences.

        Args:
            sequences: List of protein sequences
            epochs: Number of training epochs
            enable_validation: If True, split data and track validation metrics
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Epochs without improvement before stopping
            output_dir: Directory to save validation plots and metrics
        """
        print(f"🏋️ Training {self.num_models} RoBERTa models...")
        logging.info(f"🏋️ Training {self.num_models} RoBERTa models...")
        if self.mask_structural_positions:
            print("🎭 Using variable-only masking: NEVER masking structural positions (A/C)")
            logging.info("🎭 Using variable-only masking: NEVER masking structural positions (A/C)")
        else:
            print("🧬 Using standard random MLM masking")
            logging.info("🧬 Using standard random MLM masking")

        logging.info(f"Training parameters: epochs={epochs}, models={self.num_models}")
        logging.info(f"Dataset size: {len(sequences)} sequences")

        if enable_validation:
            logging.info(f"📊 Validation ENABLED: {validation_split*100:.0f}% validation split")
            logging.info(f"   Early stopping patience: {early_stopping_patience} epochs")
        else:
            logging.info("📊 Validation DISABLED: Training on full dataset")

        # Prepare datasets
        if enable_validation:
            # Split sequences for train/val
            from sklearn.model_selection import train_test_split
            train_sequences, val_sequences = train_test_split(
                sequences, test_size=validation_split, random_state=42
            )
            logging.info(f"   Train: {len(train_sequences)} sequences")
            logging.info(f"   Validation: {len(val_sequences)} sequences")

            train_dataset = self.prepare_dataset(train_sequences)
            val_dataset = self.prepare_dataset(val_sequences)
        else:
            train_dataset = self.prepare_dataset(sequences)
            val_dataset = None
            train_sequences = sequences

        # Track metrics across all models
        all_model_metrics = []

        for model_idx in range(self.num_models):
            print(f"Training model {model_idx + 1}/{self.num_models}")
            logging.info(f"Training model {model_idx + 1}/{self.num_models}")

            config = self.create_model_config()
            model = RobertaForMaskedLM(config)
            model = model.to('cuda')

            if self.mask_structural_positions:
                data_collator = self.create_variable_only_masking_collator(train_sequences)
                logging.info("Using custom variable-only masking collator")
            else:
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=True,
                    mlm_probability=0.15
                )
                logging.info("Using standard MLM data collator")

            # Configure training arguments based on validation setting
            training_args = TrainingArguments(
                output_dir=f'./roberta_protein_model_{model_idx}',
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=512,
                per_device_eval_batch_size=512 if enable_validation else None,
                save_steps=10000,
                save_total_limit=1,
                logging_steps=100,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                report_to=[],
                save_strategy="epoch" if enable_validation else "no",  # Must match eval_strategy
                eval_strategy="epoch" if enable_validation else "no",
                load_best_model_at_end=enable_validation,
                metric_for_best_model="loss" if enable_validation else None,
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if enable_validation else None,
            )

            # Add custom callback for early stopping if validation enabled
            if enable_validation:
                from transformers import EarlyStoppingCallback
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=0.0
                )
                trainer.add_callback(early_stopping)
                logging.info(f"   Added early stopping callback (patience={early_stopping_patience})")

            logging.info(f"Starting training for model {model_idx + 1}")
            train_result = trainer.train()
            logging.info(f"Completed training for model {model_idx + 1}")

            # Store training history
            if enable_validation and hasattr(trainer.state, 'log_history'):
                all_model_metrics.append({
                    'model_idx': model_idx,
                    'log_history': trainer.state.log_history,
                    'train_result': train_result
                })

            self.models.append(model)

        print("✅ Model training complete!")
        logging.info("✅ Model training complete!")

        # Save validation plots and metrics if enabled
        if enable_validation and output_dir and all_model_metrics:
            self._save_validation_plots_and_metrics(all_model_metrics, output_dir, epochs)

    def _save_validation_plots_and_metrics(self, all_model_metrics: List[Dict],
                                          output_dir: Path, max_epochs: int):
        """Save validation plots and metrics to help determine optimal epochs."""
        import json

        validation_dir = output_dir / 'validation_metrics'
        validation_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📊 Saving validation plots and metrics to {validation_dir}")

        # Extract metrics for each model
        for model_data in all_model_metrics:
            model_idx = model_data['model_idx']
            log_history = model_data['log_history']

            # Parse log history - separate train and val
            train_losses = []
            train_epochs = []
            val_losses = []

            for entry in log_history:
                if 'loss' in entry and 'epoch' in entry:  # Training loss
                    epoch = entry['epoch']
                    # Only keep one entry per epoch (the last one)
                    if epoch not in train_epochs:
                        train_epochs.append(epoch)
                        train_losses.append(entry['loss'])
                    else:
                        # Update the last entry for this epoch
                        idx = train_epochs.index(epoch)
                        train_losses[idx] = entry['loss']

                if 'eval_loss' in entry:  # Validation loss
                    val_losses.append(entry['eval_loss'])

            # Validation epochs are simply 1, 2, 3, ...
            val_epochs = list(range(1, len(val_losses) + 1))

            # Save metrics to JSON
            metrics_file = validation_dir / f'model_{model_idx}_metrics.json'
            metrics_data = {
                'model_idx': model_idx,
                'train_epochs': train_epochs,
                'train_losses': train_losses,
                'val_epochs': val_epochs,
                'val_losses': val_losses,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'best_val_loss': min(val_losses) if val_losses else None,
                'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else None
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            logging.info(f"   Model {model_idx}: Best val loss {metrics_data['best_val_loss']:.4f} at epoch {metrics_data['best_epoch']}")

        # Create consolidated plot
        self._plot_training_curves(all_model_metrics, validation_dir, max_epochs)

        # Create summary report
        self._create_validation_summary(all_model_metrics, validation_dir)

    def _plot_training_curves(self, all_model_metrics: List[Dict],
                             validation_dir: Path, max_epochs: int):
        """Create plots showing training and validation loss curves."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Collect all data
        all_train_losses = []
        all_val_losses = []
        all_epochs = []

        for model_data in all_model_metrics:
            model_idx = model_data['model_idx']
            log_history = model_data['log_history']

            # Parse metrics - separate handling for train and val
            train_losses = []
            train_epochs = []
            val_losses = []
            val_epochs = []

            for entry in log_history:
                if 'loss' in entry and 'epoch' in entry:
                    # Training loss (may be logged multiple times per epoch)
                    epoch = entry['epoch']
                    # Only keep one entry per epoch (the last one)
                    if epoch not in train_epochs:
                        train_epochs.append(epoch)
                        train_losses.append(entry['loss'])
                    else:
                        # Update the last entry for this epoch
                        idx = train_epochs.index(epoch)
                        train_losses[idx] = entry['loss']

                if 'eval_loss' in entry:
                    # Validation loss (once per epoch)
                    val_losses.append(entry['eval_loss'])
                    # Validation epochs are simple: 1, 2, 3, ...
                    val_epochs.append(len(val_losses))

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_epochs.append(train_epochs)

            # Plot individual model curves (lighter lines)
            if train_losses and train_epochs:
                axes[0].plot(train_epochs, train_losses, alpha=0.3, color='blue', linewidth=1)
            if val_losses and val_epochs:
                axes[1].plot(val_epochs, val_losses, alpha=0.3, color='orange', linewidth=1)

        # Plot average curves (bold lines)
        if all_train_losses:
            # Get max length to handle variable epoch counts (early stopping)
            max_len = max(len(losses) for losses in all_train_losses)
            avg_train_losses = []
            avg_epochs = list(range(1, max_len + 1))

            for i in range(max_len):
                epoch_losses = [losses[i] for losses in all_train_losses if i < len(losses)]
                avg_train_losses.append(np.mean(epoch_losses))

            axes[0].plot(avg_epochs, avg_train_losses, color='darkblue',
                        linewidth=3, label='Average Train Loss', marker='o')
            axes[0].set_title('Training Loss Across Epochs', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        if all_val_losses:
            max_len = max(len(losses) for losses in all_val_losses)
            avg_val_losses = []
            avg_epochs = list(range(1, max_len + 1))

            for i in range(max_len):
                epoch_losses = [losses[i] for losses in all_val_losses if i < len(losses)]
                avg_val_losses.append(np.mean(epoch_losses))

            axes[1].plot(avg_epochs, avg_val_losses, color='darkorange',
                        linewidth=3, label='Average Val Loss', marker='o')

            # Mark best epoch
            best_epoch = avg_val_losses.index(min(avg_val_losses)) + 1
            best_loss = min(avg_val_losses)
            axes[1].axvline(x=best_epoch, color='red', linestyle='--',
                          label=f'Best Epoch: {best_epoch}')
            axes[1].scatter([best_epoch], [best_loss], color='red', s=100, zorder=5)

            axes[1].set_title('Validation Loss Across Epochs', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = validation_dir / 'training_validation_curves.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"   Saved training curves to {plot_file}")

    def _create_validation_summary(self, all_model_metrics: List[Dict],
                                   validation_dir: Path):
        """Create a text summary of validation results."""
        summary_file = validation_dir / 'validation_summary.txt'

        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VALIDATION SUMMARY - OPTIMAL EPOCH DETERMINATION\n")
            f.write("="*60 + "\n\n")

            # Analyze each model
            best_epochs = []
            best_val_losses = []

            for model_data in all_model_metrics:
                model_idx = model_data['model_idx']
                log_history = model_data['log_history']

                # Extract validation losses
                val_losses = []
                for entry in log_history:
                    if 'eval_loss' in entry:
                        val_losses.append(entry['eval_loss'])

                if val_losses:
                    best_epoch = val_losses.index(min(val_losses)) + 1
                    best_loss = min(val_losses)
                    final_loss = val_losses[-1]

                    best_epochs.append(best_epoch)
                    best_val_losses.append(best_loss)

                    f.write(f"Model {model_idx}:\n")
                    f.write(f"  Best Epoch: {best_epoch}\n")
                    f.write(f"  Best Val Loss: {best_loss:.6f}\n")
                    f.write(f"  Final Val Loss: {final_loss:.6f}\n")
                    f.write(f"  Total Epochs: {len(val_losses)}\n\n")

            # Overall recommendation
            f.write("="*60 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*60 + "\n\n")

            if best_epochs:
                avg_best_epoch = np.mean(best_epochs)
                std_best_epoch = np.std(best_epochs)
                median_best_epoch = np.median(best_epochs)

                f.write(f"Average Best Epoch: {avg_best_epoch:.1f} ± {std_best_epoch:.1f}\n")
                f.write(f"Median Best Epoch: {median_best_epoch:.0f}\n")
                f.write(f"Range: {min(best_epochs)} - {max(best_epochs)}\n\n")

                recommended_epochs = int(np.ceil(median_best_epoch))
                f.write(f"RECOMMENDED EPOCHS FOR PRODUCTION: {recommended_epochs}\n\n")

                f.write("Usage:\n")
                f.write(f"  python motif_discovery_position_independent.py \\\n")
                f.write(f"    --csv your_data.csv \\\n")
                f.write(f"    --epochs {recommended_epochs} \\\n")
                f.write(f"    --output-dir production_results\n")

        logging.info(f"   Saved validation summary to {summary_file}")

        # Also log the recommendation
        if best_epochs:
            recommended = int(np.ceil(np.median(best_epochs)))
            logging.info(f"")
            logging.info(f"{'='*60}")
            logging.info(f"🎯 RECOMMENDED EPOCHS: {recommended}")
            logging.info(f"{'='*60}")
            logging.info(f"   Based on median of best validation epochs across {len(best_epochs)} models")
            logging.info(f"   Use --epochs {recommended} for production runs")

    def create_variable_only_masking_collator(self, sequences: List[str]):
        """
        ★★★ THIS IS THE KEY METHOD - PREVENTS A/C MASKING ★★★
        Create a custom data collator that only masks variable positions
        """

        class VariableOnlyMaskingCollator:
            def __init__(self, tokenizer, sequences, get_structural_mask_positions, mlm_probability=0.15):
                self.tokenizer = tokenizer
                self.sequences = sequences
                self.get_structural_mask_positions = get_structural_mask_positions
                self.mlm_probability = mlm_probability
                self.mask_token_id = tokenizer.mask_token_id

            def __call__(self, examples):
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []

                for example in examples:
                    input_ids = torch.tensor(example['input_ids'])
                    attention_mask = torch.tensor(example['attention_mask'])
                    labels = input_ids.clone()

                    # ===== KEY PART: PREVENT MASKING OF STRUCTURAL POSITIONS =====
                    # Decode sequence to identify structural positions
                    sequence_tokens = input_ids[1:-1]  # Remove <s> and </s>
                    sequence = ''.join([self.tokenizer._id_to_token.get(token_id.item(), '')
                                       for token_id in sequence_tokens])

                    # Get structural positions (A and C positions)
                    structural_positions = set(self.get_structural_mask_positions(sequence))

                    # Create variable positions (exclude structural)
                    variable_positions = []
                    for i in range(len(sequence)):
                        if i not in structural_positions:  # ← This excludes A/C from masking
                            variable_positions.append(i + 1)  # +1 for <s> token offset

                    # Apply MLM masking ONLY to variable positions
                    num_to_mask = int(len(variable_positions) * self.mlm_probability)

                    if num_to_mask > 0:
                        import random
                        positions_to_mask = random.sample(variable_positions,
                                                         min(num_to_mask, len(variable_positions)))

                        for pos in positions_to_mask:
                            if pos < len(input_ids):
                                rand = random.random()
                                if rand < 0.8:
                                    input_ids[pos] = self.mask_token_id
                                elif rand < 0.9:
                                    aa_tokens = list(range(5, 25))  # Amino acid token IDs
                                    input_ids[pos] = random.choice(aa_tokens)
                                # else: keep original token (10% case)
                    # ===== END KEY PART =====

                    # Set labels for loss calculation
                    labels.fill_(-100)  # Ignore all positions by default
                    for pos in positions_to_mask:
                        if pos < len(labels):
                            labels[pos] = example['input_ids'][pos]  # Original token

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(labels)

                return {
                    'input_ids': torch.stack(batch_input_ids),
                    'attention_mask': torch.stack(batch_attention_mask),
                    'labels': torch.stack(batch_labels)
                }

        return VariableOnlyMaskingCollator(
            self.tokenizer,
            sequences,
            self.get_structural_mask_positions,
            mlm_probability=0.15
        )

    def extract_attention_weights(self, sequences: List[str], batch_size: int = 64) -> np.ndarray:
        """Extract self-attention weights for sequences using batch processing"""
        print(f"🔍 Extracting self-attention weights with batch size {batch_size}...")
        logging.info(f"🔍 Extracting self-attention weights with batch size {batch_size}...")
        if self.mask_structural_positions:
            print("🎭 Structural positions will be masked during motif discovery (not attention extraction)")
            logging.info("🎭 Structural positions will be masked during motif discovery (not attention extraction)")

        attention_vectors = []

        # Process sequences in batches
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]

            if batch_start % (batch_size * 10) == 0:
                logging.info(f"Processing batch {batch_start//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size}")

            # Tokenize entire batch
            encoded_batch = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids_batch = encoded_batch['input_ids'].cuda()
            attention_mask_batch = encoded_batch['attention_mask'].cuda()

            # Process each model
            batch_model_attentions = []
            for model in self.models:
                model.eval()
                model = model.to('cuda')
                with torch.no_grad():
                    outputs = model(input_ids_batch,
                                  attention_mask=attention_mask_batch,
                                  output_attentions=True)
                    attentions = outputs.attentions

                    # Average across layers and heads for the entire batch
                    layer_attentions = []
                    for layer_att in attentions:
                        # layer_att shape: [batch_size, num_heads, seq_len, seq_len]
                        avg_heads = layer_att.mean(dim=1)  # Average across heads: [batch_size, seq_len, seq_len]
                        layer_attentions.append(avg_heads)

                    # Average across layers: [batch_size, seq_len, seq_len]
                    avg_attention_batch = torch.stack(layer_attentions).mean(dim=0)

                    # Extract column sum attention for each sequence in batch
                    batch_attention_scores = []
                    for seq_idx in range(len(batch_sequences)):
                        seq_len = len(batch_sequences[seq_idx])
                        seq_attention = avg_attention_batch[seq_idx]
                        # Column sum attention (how much attention each position receives)
                        attention_scores = seq_attention.sum(dim=0)[1:seq_len+1]  # Skip [CLS] token
                        batch_attention_scores.append(attention_scores.cpu().numpy())

                    batch_model_attentions.append(batch_attention_scores)

            # Average across models for this batch
            for seq_idx in range(len(batch_sequences)):
                model_attentions_for_seq = [batch_model_attentions[model_idx][seq_idx]
                                           for model_idx in range(len(self.models))]
                avg_model_attention = np.mean(model_attentions_for_seq, axis=0)

                # Pad to max length
                padded_attention = np.zeros(self.max_length - 2)
                padded_attention[:len(avg_model_attention)] = avg_model_attention
                attention_vectors.append(padded_attention)

        attention_array = np.array(attention_vectors)
        logging.info(f"Attention extraction complete. Shape: {attention_array.shape}")
        logging.info(f"Attention range: {attention_array.min():.6f} to {attention_array.max():.6f}")
        logging.info(f"Attention mean: {attention_array.mean():.6f}, std: {attention_array.std():.6f}")

        return attention_array

    def setup_analysis_directories(self, output_dir: str):
        """Create organized directory structure for comprehensive analysis"""
        base_path = Path(output_dir)

        # Create subdirectories
        subdirs = [
            'attention_analysis',
            'threshold_analysis',
            'motifs_by_method',
            'visualizations',
            'data_exports'
        ]

        for subdir in subdirs:
            (base_path / subdir).mkdir(parents=True, exist_ok=True)

        logging.info(f"📁 Created analysis directories in: {base_path.absolute()}")
        return base_path

    def _get_high_attention_sequence(self, sequence: str, attention_scores: np.ndarray,
                                   threshold: float) -> Optional[str]:
        """Convert sequence to high-attention pattern"""
        high_attention_pattern = []

        if self.mask_structural_positions:
            mask_positions = set(self.get_structural_mask_positions(sequence))
        else:
            mask_positions = set()

        for i, aa in enumerate(sequence):
            if i in mask_positions:
                high_attention_pattern.append('x')
            elif i < len(attention_scores) and attention_scores[i] > threshold:
                high_attention_pattern.append(aa)
            else:
                high_attention_pattern.append('x')

        pattern = ''.join(high_attention_pattern)

        if pattern.count('x') < len(pattern) - 1:
            return pattern
        return None

    def extract_core_motif(self, pattern: str) -> str:
        """
        Extract position-independent core motif from fixed-position pattern

        Examples:
        xxWWxxxYYxx → WW[3]YY  (WW followed by 3-gap then YY)
        xxxWWxxxYYxx → WW[3]YY (same core motif)
        xxWWxxxYYxxxxx → WW[3]YY (same core motif)
        """

        if not pattern or pattern.count('x') == len(pattern):
            return None

        # Find all conserved segments (consecutive non-x characters)
        conserved_segments = []
        current_segment = ""

        for char in pattern:
            if char != 'x':
                current_segment += char
            else:
                if current_segment:
                    conserved_segments.append(current_segment)
                    current_segment = ""

        # Don't forget the last segment
        if current_segment:
            conserved_segments.append(current_segment)

        if len(conserved_segments) < 1:
            return None

        # For single segment, just return it
        if len(conserved_segments) == 1:
            return conserved_segments[0]

        # For multiple segments, calculate gaps between them
        core_motif_parts = []
        i = 0

        for seg_idx, segment in enumerate(conserved_segments):
            # Find this segment in the pattern
            seg_start = pattern.find(segment, i)

            if seg_idx == 0:
                # First segment
                core_motif_parts.append(segment)
            else:
                # Calculate gap from end of previous segment
                prev_segment = conserved_segments[seg_idx - 1]
                prev_end = pattern.rfind(prev_segment, 0, seg_start) + len(prev_segment)
                gap_size = seg_start - prev_end

                core_motif_parts.append(f"[{gap_size}]")
                core_motif_parts.append(segment)

            i = seg_start + len(segment)

        return ''.join(core_motif_parts)

    def group_motifs_by_core(self, motif_groups: Dict) -> Dict:
        """
        Re-group motifs by their position-independent core patterns

        Returns:
            Dict[core_pattern, List[Dict]] - core patterns mapped to combined motif data
        """

        core_motif_groups = defaultdict(list)

        # Extract core motifs and group them
        for pattern, indices in motif_groups.items():
            core_motif = self.extract_core_motif(pattern)

            if core_motif:
                core_motif_groups[core_motif].append({
                    'original_pattern': pattern,
                    'indices': indices,
                    'count': len(indices)
                })

        # Combine groups with the same core motif
        combined_groups = {}

        for core_pattern, pattern_groups in core_motif_groups.items():
            # Combine all indices from patterns with same core
            all_indices = []
            original_patterns = []

            for group in pattern_groups:
                all_indices.extend(group['indices'])
                original_patterns.append(f"{group['original_pattern']}({group['count']})")

            combined_groups[core_pattern] = {
                'indices': all_indices,
                'count': len(all_indices),
                'original_patterns': original_patterns,
                'core_pattern': core_pattern
            }

        return combined_groups

    def _discover_motifs_position_independent(self, sequences: List[str], attention_vectors: np.ndarray,
                                        percentile: float, min_instances: int) -> List[Dict]:
        """
        Discover motifs using position-independent core patterns

        This addresses the issue where xxWWxxxYYxx, xxxWWxxxYYxx, and xxWWxxxYYxxxxx
        should be grouped together as the same core motif: WW[3]YY
        """

        logging.info(f"🔄 Discovering position-independent motifs with {percentile}th percentile threshold")

        # Step 1: Get initial fixed-position patterns (same as before)
        motif_groups = defaultdict(list)

        for i, seq in enumerate(sequences):
            attention = attention_vectors[i][:len(seq)]
            threshold = np.percentile(attention, percentile)

            high_attention_seq = self._get_high_attention_sequence(seq, attention_vectors[i], threshold)

            if high_attention_seq:
                motif_groups[high_attention_seq].append(i)

        logging.info(f"Initial fixed-position patterns: {len(motif_groups)}")

        # Step 2: Group by position-independent core motifs
        core_grouped = self.group_motifs_by_core(motif_groups)

        logging.info(f"After core motif grouping: {len(core_grouped)}")
        logging.info("Core motif examples:")
        for i, (core, data) in enumerate(list(core_grouped.items())[:5]):
            logging.info(f"  {core}: {data['count']} sequences from patterns: {', '.join(data['original_patterns'][:3])}")

        # Step 3: Build PSSM consensus for core motifs
        motifs = self._build_position_independent_consensus(sequences, attention_vectors,
                                                           core_grouped, min_instances)

        # Filter by minimum instances and sort
        motifs = [m for m in motifs if m['count'] >= min_instances]
        motifs.sort(key=lambda x: x['count'], reverse=True)

        return motifs

    def _build_position_independent_consensus(self, sequences: List[str], attention_vectors: np.ndarray,
                                        core_grouped: Dict, min_instances: int) -> List[Dict]:
        """
        Build consensus motifs for position-independent core patterns
        """

        motifs = []
        amino_acids = list("ARNDCQEGHILKMFPSTWYV")

        for core_pattern, group_data in core_grouped.items():
            indices = group_data['indices']

            if len(indices) < min_instances:
                continue

            # Get sequences for this core motif group
            motif_sequences = [sequences[i] for i in indices]

            # Create simple consensus from the sequences (PSSMs will be built later from assignments)
            consensus_sequence = motif_sequences[0] if motif_sequences else ''

            motif = {
                'core_pattern': core_pattern,
                'pattern': f"CORE:{core_pattern}",  # Mark as position-independent
                'consensus_sequence': consensus_sequence,
                'sequences': motif_sequences,
                'indices': indices,
                'count': len(indices),
                'avg_attention': np.mean([attention_vectors[i] for i in indices], axis=0),
                'pattern_type': 'position_independent',
                'original_patterns': group_data['original_patterns']
            }

            motifs.append(motif)

        return motifs

    def _calculate_avg_conservation(self, length_combinations: Dict) -> float:
        """Calculate average conservation score across all length combinations"""
        if not length_combinations:
            return 0.0

        total_conservation = 0.0
        total_sequences = 0

        for combo_key, combo_data in length_combinations.items():
            # Weight by number of sequences in each combination
            seq_count = combo_data.get('sequence_count', 0)
            avg_info = combo_data.get('avg_information_content', 0)
            total_conservation += avg_info * seq_count
            total_sequences += seq_count

        return total_conservation / total_sequences if total_sequences > 0 else 0.0

    def _get_loop_lengths(self, sequence):
        """Extract loop1 and loop2 lengths from AC...C...CA structure"""

        if not (sequence.startswith('AC') and sequence.endswith('A')):
            raise ValueError(f"Sequence doesn't match AC...A pattern: {sequence}")

        # Find all cysteine positions
        c_positions = [i for i, aa in enumerate(sequence) if aa == 'C']

        if len(c_positions) < 3:
            raise ValueError(f"Expected 3 cysteines, found {len(c_positions)}: {sequence}")

        # Structure: A-C-[loop1]-C-[loop2]-C-A
        loop1_length = c_positions[1] - 2  # From position 2 to middle C
        loop2_length = c_positions[2] - c_positions[1] - 1  # Between middle C and final C

        return loop1_length, loop2_length

    def _group_motifs_by_loop_lengths(self, sequences: List[str], indices: List[int]) -> Dict:
        """Group motif sequences by their loop length combinations"""

        length_groups = defaultdict(list)

        for idx in indices:
            seq = sequences[idx]
            try:
                loop1_len, loop2_len = self._get_loop_lengths(seq)
                length_key = f"{loop1_len}x{loop2_len}"
                length_groups[length_key].append({
                    'index': idx,
                    'sequence': seq,
                    'loop1_length': loop1_len,
                    'loop2_length': loop2_len
                })
            except ValueError as e:
                logging.warning(f"Skipping invalid sequence {idx}: {e}")
                continue

        return length_groups

    def _build_length_stratified_pssms(self, sequences: List[str], attention_vectors: np.ndarray,
                                      motif_indices: List[int], min_instances: int = 5) -> Dict:
        """Build separate PSSMs for each loop length combination"""

        # Group by loop lengths
        length_groups = self._group_motifs_by_loop_lengths(sequences, motif_indices)

        length_specific_pssms = {}
        amino_acids = list("ARNDCQEGHILKMFPSTWYV")

        logging.info(f"Found {len(length_groups)} different loop length combinations:")
        for length_key, group_data in length_groups.items():
            logging.info(f"  {length_key}: {len(group_data)} sequences")

        for length_key, group_data in length_groups.items():
            if len(group_data) < min_instances:
                logging.info(f"Skipping {length_key}: only {len(group_data)} sequences (need {min_instances})")
                continue

            # Extract sequences for this length combination
            group_sequences = [item['sequence'] for item in group_data]
            group_indices = [item['index'] for item in group_data]

            # All sequences in this group have identical length - perfect for standard PSSM!
            seq_length = len(group_sequences[0])

            # Build standard PSSM (no alignment issues!)
            pssm_counts = np.zeros((seq_length, len(amino_acids)))
            position_attention_scores = np.zeros(seq_length)

            print(f"DEBUG: Processing length combination: {length_key}")
            print(f"DEBUG: Number of sequences in group: {len(group_data)}")
            print(f"DEBUG: Expected sequence length: {seq_length}")
            print(f"DEBUG: PSSM matrix shape: {pssm_counts.shape}")
            print(f"DEBUG: First few sequence lengths: {[len(item['sequence']) for item in group_data[:5]]}")

            for item in group_data:
                seq = item['sequence']
                print(f"DEBUG: Processing sequence of length {len(seq)}: '{seq}'")
                attention = attention_vectors[item['index']][:len(seq)]

                for pos, aa in enumerate(seq):
                    print(f"DEBUG: pos={pos}, aa={aa}, seq_length={len(seq)}")
                    if pos >= pssm_counts.shape[0]:
                        print(f"ERROR: pos {pos} >= matrix size {pssm_counts.shape[0]}")
                        break
                    if aa in amino_acids:
                        aa_idx = amino_acids.index(aa)
                        pssm_counts[pos][aa_idx] += 1
                        position_attention_scores[pos] += attention[pos] if pos < len(attention) else 0

            # Convert to frequencies
            pssm_frequencies = pssm_counts / len(group_sequences)
            position_attention_scores /= len(group_sequences)

            # Calculate information content
            background_freq = 1.0 / len(amino_acids)
            information_content = np.zeros(seq_length)

            for pos in range(seq_length):
                for aa_idx in range(len(amino_acids)):
                    freq = pssm_frequencies[pos][aa_idx]
                    if freq > 0:
                        information_content[pos] += freq * np.log2(freq / background_freq)

            # Build consensus
            consensus_sequence = ""
            consensus_pattern = ""

            for pos in range(seq_length):
                most_frequent_aa_idx = np.argmax(pssm_frequencies[pos])
                most_frequent_aa = amino_acids[most_frequent_aa_idx]
                frequency = pssm_frequencies[pos][most_frequent_aa_idx]

                consensus_sequence += most_frequent_aa

                if frequency >= 0.8:
                    consensus_pattern += most_frequent_aa
                elif frequency >= 0.6:
                    consensus_pattern += most_frequent_aa.lower()
                elif information_content[pos] > 1.0:
                    consensus_pattern += '+'
                else:
                    consensus_pattern += '.'

            # Create PSSM matrix for export
            pssm_matrix = []
            for pos in range(seq_length):
                pos_data = {
                    'position': pos,
                    'information_content': float(information_content[pos]),
                    'avg_attention': float(position_attention_scores[pos]),
                    'amino_acid_frequencies': {
                        aa: float(pssm_frequencies[pos][i])
                        for i, aa in enumerate(amino_acids)
                        if pssm_frequencies[pos][i] > 0.01
                    }
                }
                pssm_matrix.append(pos_data)

            # Store length-specific analysis
            loop1_len, loop2_len = length_key.split('x')
            length_specific_pssms[length_key] = {
                'length_combination': length_key,
                'loop1_length': int(loop1_len),
                'loop2_length': int(loop2_len),
                'sequence_count': len(group_sequences),
                'consensus_sequence': consensus_sequence,
                'consensus_pattern': consensus_pattern,
                'sequences': group_sequences,
                'indices': group_indices,
                'pssm_matrix': pssm_matrix,
                'pssm_frequencies': pssm_frequencies.tolist(),
                'information_content': information_content.tolist(),
                'total_information_content': float(np.sum(information_content)),
                'avg_information_content': float(np.mean(information_content)),
                'structural_annotation': {
                    'positions_0_1': 'AC (N-terminal anchor)',
                    'positions_2_to': f'Loop1 ({loop1_len} residues)',
                    'position': f'{2 + int(loop1_len)} (Middle C)',
                    'positions_to': f'Loop2 ({loop2_len} residues)',
                    'positions_final': 'CA (C-terminal anchor)'
                }
            }

        logging.info(f"Built {len(length_specific_pssms)} length-stratified PSSMs")
        return length_specific_pssms

    def _test_position_independent_approaches(self, sequences: List[str], attention_vectors: np.ndarray,
                                        base_path: Path, min_instances: int, consolidate: bool = True) -> Dict:
        """Test position-independent motif discovery approaches with optional consolidation (enabled by default)"""

        percentiles = [65, 70, 75]
        results = {}

        for percentile in percentiles:
            logging.info(f"  Testing position-independent with percentile={percentile}%")

            # Use position-independent discovery
            motifs = self._discover_motifs_position_independent(
                sequences, attention_vectors, percentile, min_instances
            )

            # If consolidation is enabled, consolidate motifs before saving
            if consolidate and motifs:
                logging.info(f"  🔀 Consolidating motifs for percentile={percentile}%...")

                # Build hierarchy from motif patterns
                import pandas as pd
                df_temp = pd.DataFrame([{
                    'core_pattern': m.get('core_pattern', m.get('pattern', '')),
                    'total_sequences': m.get('count', 0)
                } for m in motifs])

                hierarchy, parent_map = find_motif_hierarchy(df_temp)
                logging.info(f"    Found {len(hierarchy)} parent motifs with children")

                # Consolidate motif dictionaries
                motifs = consolidate_motif_dictionaries(motifs, hierarchy)

                # Rebuild PSSMs for consolidated motifs
                logging.info(f"    Rebuilding PSSMs for {len(motifs)} consolidated motifs...")
                motifs = self.rebuild_pssms_for_consolidated_motifs(motifs, sequences)

                # Export sequences for each consolidated motif
                logging.info(f"    Exporting sequences for consolidated motifs...")
                export_sequences_for_motifs(motifs, sequences, base_path)

                # Export CSV with sequences and their detected motifs
                logging.info(f"    Exporting CSV with sequences and motifs...")
                export_sequences_with_motifs_csv(motifs, sequences, base_path,
                                                  method_name='position_independent',
                                                  percentile=percentile,
                                                  consolidated=True)

                logging.info(f"  ✓ Consolidation complete for percentile={percentile}%")

            result = {
                'percentile': percentile,
                'approach': 'position_independent',
                'motifs': motifs,
                'num_motifs': len(motifs),
                'total_sequences': sum(m['count'] for m in motifs),
                'avg_motif_size': np.mean([m['count'] for m in motifs]) if motifs else 0,
                'avg_conservation': np.mean([m.get('conservation_score', 0) for m in motifs]) if motifs else 0
            }

            results[f'pos_indep_p{percentile}'] = result

            # Save motifs
            if motifs:
                csv_filename = f'position_independent_p{percentile}_{"consolidated_" if consolidate else ""}motifs.csv'
                self._save_position_independent_motifs(motifs,
                    base_path / 'motifs_by_method' / csv_filename)

            # Log results
            if motifs:
                logging.info(f"  Found {len(motifs)} position-independent motifs, top 3:")
                for i, motif in enumerate(motifs[:3]):
                    original_patterns = ', '.join(motif['original_patterns'][:2])
                    logging.info(f"    {i+1}. {motif['core_pattern']} ({motif['count']} seqs from: {original_patterns})")

        return results

    def _save_position_independent_motifs(self, motifs: List[Dict], filepath: Path):
        """Save position-independent motifs to CSV"""
        if not motifs:
            return

        import csv

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            headers = [
                'rank', 'core_pattern', 'consensus_sequence', 'total_sequences',
                'original_count', 'num_children',
                'num_length_combinations', 'most_common_combination',
                'conservation_score', 'original_patterns_combined',
                'child_motifs', 'example_sequences'
            ]
            writer.writerow(headers)

            for i, motif in enumerate(motifs):
                row = [
                    i + 1,
                    motif.get('core_pattern', ''),
                    motif.get('consensus_sequence', ''),
                    motif.get('count', 0),  # Internal uses 'count', CSV shows as 'total_sequences'
                    motif.get('original_count', motif.get('count', 0)),  # Count before consolidation
                    motif.get('num_children', 0),  # Number of children merged
                    motif.get('num_length_combinations', 0),
                    motif.get('most_common_combination', 'N/A'),
                    motif.get('conservation_score', 0),
                    '; '.join(motif.get('original_patterns', [])[:5]),
                    '; '.join(motif.get('child_motifs', [])[:10]),  # List of child motif names
                    '; '.join(motif.get('sequences', [])[:5])
                ]
                writer.writerow(row)

        logging.info(f"Saved position-independent motifs to: {filepath}")

    def rebuild_pssms_for_consolidated_motifs(self, motifs: List[Dict], sequences: List[str]) -> List[Dict]:
        """Rebuild length-stratified PSSMs for consolidated motifs using their combined sequences"""
        logging.info(f"Rebuilding PSSMs for {len(motifs)} consolidated motifs...")

        for motif in motifs:
            if not motif.get('needs_pssm_rebuild', False) and motif.get('length_combinations'):
                continue  # Already has PSSMs

            # Get all sequences for this consolidated motif
            motif_sequences = [sequences[i] for i in motif['indices']]

            # Create dummy attention vectors (not needed for PSSM building)
            dummy_attention = np.ones((len(sequences), max(len(s) for s in sequences)))

            # Rebuild length-stratified PSSMs from consolidated sequences
            # This calls the existing _build_length_stratified_pssms method
            length_combinations = self._build_length_stratified_pssms(
                sequences,
                dummy_attention,
                motif['indices'],
                min_instances=5
            )

            # Update motif with new PSSM data
            motif['length_combinations'] = length_combinations

            # Calculate summary statistics
            if length_combinations:
                most_common_combo = max(length_combinations.items(),
                                       key=lambda x: x[1]['sequence_count'])
                motif['most_common_combination'] = most_common_combo[0]
                motif['consensus_sequence'] = most_common_combo[1]['consensus_sequence']
                motif['consensus_pattern'] = most_common_combo[1]['consensus_pattern']
                motif['num_length_combinations'] = len(length_combinations)
                motif['total_information_content'] = sum(
                    combo['total_information_content']
                    for combo in length_combinations.values()
                )
                motif['avg_information_per_combination'] = motif['total_information_content'] / len(length_combinations)
                motif['highly_conserved_positions'] = sum(
                    len([pos for pos in combo['information_content'] if pos > 2.0])
                    for combo in length_combinations.values()
                )

            motif['needs_pssm_rebuild'] = False

        logging.info("PSSM rebuild complete")
        return motifs

    def create_pssm_heatmaps_UPDATED(self, motifs: List[Dict], attention_matrices: List[np.ndarray], output_dir: Path, top_n: int = 10):
        """Create PSSM heatmaps for top motifs with length-stratified data"""

        # Filter motifs that have length-stratified PSSM data
        motifs_with_pssm = [m for m in motifs if 'length_combinations' in m and m['length_combinations']]
        if not motifs_with_pssm:
            logging.info("No motifs with length-stratified PSSM data found for heatmap generation")
            return

        # Take top N motifs by total information content
        top_motifs = motifs_with_pssm[:min(top_n, len(motifs_with_pssm))]

        # Create logo plots for each motif
        self._create_length_stratified_logo_plots(top_motifs, output_dir, top_n=5)

        # Create comparison overview
        if len(top_motifs) > 1:
            self._create_logo_plots_comparison_overview(top_motifs, output_dir)

        logging.info(f"Created length-stratified PSSM heatmaps AND logo plots for {len(top_motifs)} motifs in {output_dir}")

    def _create_length_stratified_logo_plots(self, motifs: List[Dict], output_dir: Path, top_n: int = 5):
        """Create sequence logo-style plots for length-stratified motifs"""

        top_motifs = motifs[:min(top_n, len(motifs))]

        for motif_idx, motif in enumerate(top_motifs):
            core_pattern = motif.get('core_pattern', f'Motif_{motif_idx+1}')
            length_combinations = motif.get('length_combinations', {})

            if not length_combinations:
                continue

            # Create logo plots for each length combination of this motif
            self._create_logo_plots_for_motif_length_combinations(
                motif, motif_idx, output_dir
            )

        # Create overview logo comparison
        self._create_logo_plots_comparison_overview(top_motifs, output_dir)

        logging.info(f"Created sequence logo plots for {len(top_motifs)} motifs")

    def _create_logo_plots_for_motif_length_combinations(self, motif: Dict, motif_idx: int, output_dir: Path):
        """Create logo plots for all length combinations of a single motif"""

        core_pattern = motif.get('core_pattern', f'Motif_{motif_idx+1}')
        length_combinations = motif.get('length_combinations', {})

        if not length_combinations:
            return

        n_combinations = len(length_combinations)
        fig, axes = plt.subplots(n_combinations, 1, figsize=(14, n_combinations * 4))

        if n_combinations == 1:
            axes = [axes]

        amino_acids = list("ARNDCQEGHILKMFPSTWYV")
        # Create a nice color palette for amino acids
        colors = plt.cm.Set3(np.linspace(0, 1, len(amino_acids)))

        for combo_idx, (length_key, pssm_data) in enumerate(length_combinations.items()):
            pssm_matrix = pssm_data['pssm_matrix']
            consensus = pssm_data['consensus_pattern']
            count = pssm_data['sequence_count']
            loop1_len = pssm_data['loop1_length']
            loop2_len = pssm_data['loop2_length']

            ax = axes[combo_idx]
            seq_length = len(pssm_matrix)

            # For each position, stack amino acids by frequency * information content
            for pos_idx, pos_data in enumerate(pssm_matrix):
                aa_freqs = pos_data.get('amino_acid_frequencies', {})
                info_content = pos_data.get('information_content', 0)

                # Sort amino acids by frequency
                sorted_aas = sorted(aa_freqs.items(), key=lambda x: x[1], reverse=True)

                y_offset = 0
                for aa, freq in sorted_aas:
                    if freq > 0.01:  # Only show AAs with >1% frequency
                        height = freq * info_content  # Height scaled by information content
                        aa_idx = amino_acids.index(aa)

                        # Draw colored rectangle
                        rect = plt.Rectangle((pos_idx - 0.4, y_offset), 0.8, height,
                                           facecolor=colors[aa_idx], alpha=0.8,
                                           edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)

                        # Add amino acid letter
                        if height > 0.1:  # Only add text if bar is tall enough
                            ax.text(pos_idx, y_offset + height/2, aa,
                                   ha='center', va='center', fontweight='bold',
                                   fontsize=12, color='white')

                        y_offset += height

            # Add structural region backgrounds
            self._add_structural_region_backgrounds(ax, seq_length, loop1_len, loop2_len)

            ax.set_xlim(-0.5, seq_length - 0.5)
            max_height = max(2.0, max([pos_data.get('information_content', 0) for pos_data in pssm_matrix]) * 1.1)
            ax.set_ylim(0, max_height)
            ax.set_xlabel('Position')
            ax.set_ylabel('Information (bits)')
            ax.set_title(f'{length_key} peptides: {consensus} (n={count})')
            ax.set_xticks(range(seq_length))

            # Add structural annotations
            structure_labels = []
            for pos in range(seq_length):
                if pos < 2:
                    structure_labels.append(f'AC\n{pos}')
                elif pos < 2 + loop1_len:
                    structure_labels.append(f'L1\n{pos}')
                elif pos == 2 + loop1_len:
                    structure_labels.append(f'C\n{pos}')
                elif pos < 2 + loop1_len + 1 + loop2_len:
                    structure_labels.append(f'L2\n{pos}')
                else:
                    structure_labels.append(f'CA\n{pos}')

            ax.set_xticklabels(structure_labels, fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Sequence Logos: {core_pattern}', fontsize=16)
        plt.tight_layout()

        # Save individual motif logo plots
        safe_pattern = core_pattern.replace('[', '').replace(']', '').replace(':', '_')
        filename = f'motif_{motif_idx+1:02d}_{safe_pattern}_logos.png'
        plt.savefig(output_dir / 'visualizations' / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def build_pssms_from_assignments(self, motifs: List[Dict], sequences: List[str], output_dir: Path) -> List[Dict]:
        """Build length-stratified PSSMs from motif assignments and create visualizations"""
        logging.info(f"Building PSSMs from motif assignments for {len(motifs)} motifs...")

        # Create dummy attention vectors (not needed for PSSM building)
        dummy_attention = np.ones((len(sequences), max(len(s) for s in sequences)))

        for motif in motifs:
            core_pattern = motif.get('core_pattern', '')
            motif_indices = motif.get('indices', [])

            if not motif_indices:
                logging.warning(f"Motif {core_pattern} has no assigned sequences")
                continue

            logging.info(f"Processing motif {core_pattern} with {len(motif_indices)} sequences")

            # Build length-stratified PSSMs for this motif
            length_combinations = self._build_length_stratified_pssms(
                sequences,
                dummy_attention,
                motif_indices,
                min_instances=5
            )

            # Store PSSM data in motif
            motif['length_combinations'] = length_combinations
            motif['num_length_combinations'] = len(length_combinations)

            if length_combinations:
                # Get most common combination
                most_common_combo = max(length_combinations.items(),
                                       key=lambda x: x[1]['sequence_count'])
                motif['most_common_combination'] = most_common_combo[0]
                motif['consensus_sequence'] = most_common_combo[1]['consensus_sequence']
                motif['consensus_pattern'] = most_common_combo[1]['consensus_pattern']

                # Calculate conservation score
                motif['conservation_score'] = self._calculate_avg_conservation(length_combinations)

                # Create individual logo plots for each loop length combination
                self._create_individual_logo_plots_for_motif(motif, output_dir)

                # Create detailed three-panel PSSM plots for each loop length combination
                self._create_detailed_pssm_plots_for_motif(motif, output_dir)
            else:
                logging.warning(f"No valid length combinations for motif {core_pattern}")

        logging.info("PSSM building and visualization creation complete")
        return motifs

    def _create_individual_logo_plots_for_motif(self, motif: Dict, output_dir: Path):
        """Create separate logo plot for each motif-loop length combination"""
        core_pattern = motif.get('core_pattern', '')
        length_combinations = motif.get('length_combinations', {})

        if not length_combinations:
            return

        # Ensure visualizations directory exists
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename base
        safe_pattern = core_pattern.replace('[', '').replace(']', '').replace(':', '_')

        amino_acids = list("ARNDCQEGHILKMFPSTWYV")
        # Create color map for amino acids (each gets unique color)
        colors = plt.cm.Set3(np.linspace(0, 1, len(amino_acids)))

        for length_key, combo_data in length_combinations.items():
            # Create a single logo plot for this combination
            pssm_frequencies = np.array(combo_data['pssm_frequencies'])
            consensus_sequence = combo_data['consensus_sequence']
            sequence_count = combo_data['sequence_count']
            information_content = np.array(combo_data['information_content'])

            # Parse loop lengths from length_key (e.g., "4x4" -> loop1=4, loop2=4)
            try:
                loop1_len, loop2_len = map(int, length_key.split('x'))
            except ValueError:
                logging.warning(f"Invalid length key format: {length_key}")
                continue

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 4))

            seq_length = pssm_frequencies.shape[0]

            # Add structural region backgrounds
            self._add_structural_region_backgrounds(ax, seq_length, loop1_len, loop2_len)

            # Plot sequence logo with colored rectangles
            for pos in range(seq_length):
                # Get amino acid frequencies at this position
                aa_freqs = {amino_acids[i]: pssm_frequencies[pos][i]
                           for i in range(len(amino_acids))}

                # Sort by frequency descending (most frequent on top)
                sorted_aas = sorted(aa_freqs.items(), key=lambda x: x[1], reverse=True)

                y_offset = 0
                for aa, freq in sorted_aas:
                    if freq > 0.01:  # Only show AAs with >1% frequency
                        height = freq * information_content[pos]

                        # Only draw if height is significant
                        if height > 0.1:
                            aa_idx = amino_acids.index(aa)

                            # Draw colored rectangle
                            rect = plt.Rectangle((pos - 0.4, y_offset), 0.8, height,
                                               facecolor=colors[aa_idx], alpha=0.8,
                                               edgecolor='black', linewidth=0.5)
                            ax.add_patch(rect)

                            # Add amino acid letter in white on colored background
                            ax.text(pos, y_offset + height/2, aa,
                                   ha='center', va='center', fontweight='bold',
                                   fontsize=12, color='white')

                        y_offset += height

            # Add structural annotations to x-axis
            structure_labels = []
            for pos in range(seq_length):
                if pos < 2:
                    structure_labels.append(f'AC\n{pos}')
                elif pos < 2 + loop1_len:
                    structure_labels.append(f'L1\n{pos}')
                elif pos == 2 + loop1_len:
                    structure_labels.append(f'C\n{pos}')
                elif pos < 2 + loop1_len + 1 + loop2_len:
                    structure_labels.append(f'L2\n{pos}')
                else:
                    structure_labels.append(f'CA\n{pos}')

            ax.set_xticks(range(seq_length))
            ax.set_xticklabels(structure_labels, fontsize=9)
            ax.set_xlim(-0.5, seq_length - 0.5)
            ax.set_ylim(0, max(2.5, np.max(information_content) * 1.1))
            ax.set_ylabel('Information Content (bits)', fontsize=12)
            ax.set_title(f'{core_pattern} - Loop Lengths: {length_key} - {sequence_count} sequences',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Save individual plot
            filename = f'{safe_pattern}_{length_key}_logo.png'
            plt.savefig(output_dir / 'visualizations' / filename, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Created logo plot: {filename}")

    def _format_sequence_with_structure(self, sequence: str, loop1_len: int, loop2_len: int) -> str:
        """Format sequence string to highlight structural regions with separators"""
        if len(sequence) < 4:
            return sequence

        # Structure: A-C-[loop1]-C-[loop2]-C-A
        ac_start = sequence[:2]
        loop1 = sequence[2:2+loop1_len]
        middle_c = sequence[2+loop1_len] if 2+loop1_len < len(sequence) else ''
        loop2_start = 2 + loop1_len + 1
        loop2_end = loop2_start + loop2_len
        loop2 = sequence[loop2_start:loop2_end] if loop2_end <= len(sequence) else ''
        ca_end = sequence[-2:] if len(sequence) >= 2 else ''

        return f"{ac_start}│{loop1}│{middle_c}│{loop2}│{ca_end}"

    def _create_detailed_pssm_plots_for_motif(self, motif: Dict, output_dir: Path):
        """Create three-panel detailed PSSM plots for each motif-loop length combination"""
        core_pattern = motif.get('core_pattern', '')
        length_combinations = motif.get('length_combinations', {})

        if not length_combinations:
            return

        # Ensure visualizations directory exists
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename base
        safe_pattern = core_pattern.replace('[', '').replace(']', '').replace(':', '_')

        amino_acids = list("ARNDCQEGHILKMFPSTWYV")

        for length_key, combo_data in length_combinations.items():
            pssm_frequencies = np.array(combo_data['pssm_frequencies'])
            information_content = np.array(combo_data['information_content'])
            sequence_count = combo_data['sequence_count']
            consensus_sequence = combo_data['consensus_sequence']
            consensus_pattern = combo_data['consensus_pattern']

            # Get example sequences (first 5)
            motif_sequences = motif.get('sequences', [])
            # Filter sequences that match this loop length combination
            try:
                loop1_len, loop2_len = map(int, length_key.split('x'))
            except ValueError:
                logging.warning(f"Invalid length key format: {length_key}")
                continue

            example_seqs = []
            for seq in motif_sequences:
                try:
                    seq_loop1, seq_loop2 = self._get_loop_lengths(seq)
                    if seq_loop1 == loop1_len and seq_loop2 == loop2_len:
                        example_seqs.append(seq)
                        if len(example_seqs) >= 5:
                            break
                except ValueError:
                    continue

            seq_length = pssm_frequencies.shape[0]

            # Create figure with 3 panels
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(12, seq_length * 0.6), 12))

            # PANEL 1: Frequency heatmap
            # Build frequency matrix (amino_acids x positions)
            freq_matrix = np.zeros((len(amino_acids), seq_length))
            for pos in range(seq_length):
                for aa_idx, aa in enumerate(amino_acids):
                    freq_matrix[aa_idx, pos] = pssm_frequencies[pos][aa_idx]

            im = ax1.imshow(freq_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
            ax1.set_yticks(range(len(amino_acids)))
            ax1.set_yticklabels(amino_acids)
            ax1.set_ylabel('Amino Acid', fontsize=11)

            # Add structural annotations to x-axis
            structure_labels = []
            for pos in range(seq_length):
                if pos < 2:
                    structure_labels.append(f'{pos}\nAC')
                elif pos < 2 + loop1_len:
                    structure_labels.append(f'{pos}\nL1')
                elif pos == 2 + loop1_len:
                    structure_labels.append(f'{pos}\nC')
                elif pos < 2 + loop1_len + 1 + loop2_len:
                    structure_labels.append(f'{pos}\nL2')
                else:
                    structure_labels.append(f'{pos}\nCA')

            ax1.set_xticks(range(seq_length))
            ax1.set_xticklabels(structure_labels, fontsize=8)
            ax1.set_title(f'PSSM: {core_pattern} - {length_key} (Loop1={loop1_len}, Loop2={loop2_len})\n'
                         f'Consensus: {consensus_pattern} (n={sequence_count})',
                         fontsize=12, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Frequency', fontsize=10)

            # PANEL 2: Information content bar plot
            bars = ax2.bar(range(seq_length), information_content, color='steelblue', alpha=0.7)
            ax2.set_ylabel('Information Content (bits)', fontsize=11)
            ax2.set_title('Information Content per Position', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(range(seq_length))
            ax2.set_xticklabels(structure_labels, fontsize=8)

            # Add consensus amino acids on bars
            for i, (bar, ic) in enumerate(zip(bars, information_content)):
                max_aa_idx = np.argmax(pssm_frequencies[i])
                max_freq = pssm_frequencies[i][max_aa_idx]
                if max_freq > 0.1:  # Only show if >10% frequency
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{amino_acids[max_aa_idx]}\n{max_freq:.2f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

            # PANEL 3: Example sequences
            ax3.text(0.05, 0.95, f'Example sequences ({length_key} peptides):',
                    transform=ax3.transAxes, fontweight='bold', fontsize=11)

            for i, seq in enumerate(example_seqs[:5]):
                y_pos = 0.80 - (i * 0.15)
                formatted_seq = self._format_sequence_with_structure(seq, loop1_len, loop2_len)
                ax3.text(0.05, y_pos, f'{i+1}. {formatted_seq}',
                        transform=ax3.transAxes, fontfamily='monospace', fontsize=10)

            if not example_seqs:
                ax3.text(0.05, 0.80, 'No example sequences available',
                        transform=ax3.transAxes, fontfamily='monospace', fontsize=10, style='italic')

            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')

            plt.tight_layout()

            # Save plot
            filename = f'{safe_pattern}_{length_key}_detailed_pssm.png'
            plt.savefig(output_dir / 'visualizations' / filename, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Created detailed PSSM plot: {filename}")

    def _add_structural_region_backgrounds(self, ax, seq_length: int, loop1_len: int, loop2_len: int):
        """Add subtle background colors to highlight structural regions"""

        # Define regions with different background colors
        regions = [
            {'start': 0, 'end': 2, 'color': 'lightblue', 'alpha': 0.2, 'label': 'AC'},
            {'start': 2, 'end': 2 + loop1_len, 'color': 'lightgreen', 'alpha': 0.15, 'label': 'Loop1'},
            {'start': 2 + loop1_len, 'end': 2 + loop1_len + 1, 'color': 'lightblue', 'alpha': 0.2, 'label': 'C'},
            {'start': 2 + loop1_len + 1, 'end': 2 + loop1_len + 1 + loop2_len, 'color': 'lightcoral', 'alpha': 0.15, 'label': 'Loop2'},
            {'start': seq_length - 2, 'end': seq_length, 'color': 'lightblue', 'alpha': 0.2, 'label': 'CA'}
        ]

        for region in regions:
            if region['end'] <= seq_length:
                ax.axvspan(region['start'] - 0.5, region['end'] - 0.5,
                          color=region['color'], alpha=region['alpha'])

    def _convert_pattern_to_regex(self, core_pattern: str) -> str:
        """
        Convert core pattern to regex for matching.
        Examples:
            'WPG[1]R' → 'WPG.R'
            'P[1]W[2]R' → 'P.W..R'
            'GPG' → 'GPG'
        """
        import re
        # Replace [n] with n dots (. matches any amino acid)
        regex = re.sub(r'\[(\d+)\]', lambda m: '.' * int(m.group(1)), core_pattern)
        return regex

    def _sequence_matches_pattern(self, sequence: str, core_pattern: str) -> bool:
        """
        Check if sequence exactly matches the core pattern.
        Returns True if pattern found anywhere in sequence.

        Args:
            sequence: Full amino acid sequence
            core_pattern: Core pattern like 'WPG[1]R'

        Returns:
            True if pattern matches, False otherwise
        """
        import re
        regex = self._convert_pattern_to_regex(core_pattern)
        return re.search(regex, sequence) is not None

    def _find_most_specific_motif(self, matched_motifs: List[str]) -> str:
        """
        When sequence matches multiple motifs, return most specific (longest).
        Used for primary_motif assignment.

        Args:
            matched_motifs: List of core patterns that matched

        Returns:
            The longest (most specific) core pattern, or None if list is empty
        """
        if not matched_motifs:
            return None
        return max(matched_motifs, key=len)

    def assign_all_motifs_to_sequences(self, motifs: List[Dict], sequences: List[str]) -> pd.DataFrame:
        """
        Scan ALL sequences for ALL motifs to create multi-label assignment.
        This is different from attention-based discovery which only assigns based on high attention.

        Args:
            motifs: List of discovered and consolidated motifs (each has 'core_pattern')
            sequences: All sequences to scan

        Returns:
            DataFrame with columns:
                - sequence_index: Index in original sequences list
                - sequence: The amino acid sequence
                - motifs: List of all matching motif core patterns
                - num_motifs: Count of matching motifs
                - primary_motif: Most specific (longest) matching motif
                - loop_lengths: Loop length combination (e.g., "4x4")
        """
        logging.info(f"Scanning {len(sequences)} sequences for {len(motifs)} motifs...")

        assignments = []

        for seq_idx, sequence in enumerate(sequences):
            # Find all motifs that match this sequence
            matched_motifs = []

            for motif in motifs:
                core_pattern = motif.get('core_pattern', '')
                if core_pattern and self._sequence_matches_pattern(sequence, core_pattern):
                    matched_motifs.append(core_pattern)

            # Determine primary motif (most specific / longest)
            primary_motif = self._find_most_specific_motif(matched_motifs)

            # Get loop lengths
            loop1_len, loop2_len = self._get_loop_lengths(sequence)
            loop_lengths = f"{loop1_len}x{loop2_len}" if loop1_len and loop2_len else "unknown"

            assignments.append({
                'sequence_index': seq_idx,
                'sequence': sequence,
                'motifs': matched_motifs,
                'num_motifs': len(matched_motifs),
                'primary_motif': primary_motif if primary_motif else 'unassigned',
                'loop_lengths': loop_lengths
            })

        df = pd.DataFrame(assignments)

        # Log summary statistics
        total_sequences = len(df)
        assigned_sequences = len(df[df['primary_motif'] != 'unassigned'])
        multi_motif_sequences = len(df[df['num_motifs'] > 1])

        logging.info(f"  Total sequences: {total_sequences}")
        logging.info(f"  Assigned sequences: {assigned_sequences} ({assigned_sequences/total_sequences*100:.1f}%)")
        logging.info(f"  Unassigned sequences: {total_sequences - assigned_sequences}")
        logging.info(f"  Sequences with multiple motifs: {multi_motif_sequences} ({multi_motif_sequences/total_sequences*100:.1f}%)")
        logging.info(f"  Average motifs per sequence: {df['num_motifs'].mean():.2f}")

        return df

    def calculate_motif_cooccurrence_matrix(self, assignments_df: pd.DataFrame, motifs: List[Dict],
                                           min_cooccurrence: int = 2) -> pd.DataFrame:
        """
        Calculate co-occurrence matrix showing how many sequences each pair of motifs shares.

        OPTIMIZED VERSION: Uses vectorized matrix multiplication instead of triple-nested loops.
        Expected speedup: 60-600x faster for typical datasets.

        Args:
            assignments_df: DataFrame from assign_all_motifs_to_sequences()
            motifs: List of motifs (to get ordered list of core patterns)
            min_cooccurrence: Minimum number of shared sequences to consider (default: 2)

        Returns:
            DataFrame: NxN matrix where cell [i,j] = number of sequences containing both motif i and j
        """
        logging.info(f"Calculating motif co-occurrence matrix (optimized)...")

        # Get ordered list of motif core patterns
        motif_patterns = [m.get('core_pattern', '') for m in motifs]

        # Explode motifs column and filter to relevant patterns
        exploded = assignments_df[['sequence_index', 'motifs']].explode('motifs').dropna()
        exploded = exploded[exploded['motifs'].isin(motif_patterns)]

        if exploded.empty:
            # Return empty matrix if no data
            n_motifs = len(motif_patterns)
            return pd.DataFrame(
                np.zeros((n_motifs, n_motifs), dtype=int),
                index=motif_patterns,
                columns=motif_patterns
            )

        # Create binary presence matrix using crosstab (MUCH faster than loops)
        presence = pd.crosstab(exploded['sequence_index'], exploded['motifs']).astype(np.uint8)
        presence = presence.reindex(columns=motif_patterns, fill_value=0)

        # Matrix multiplication for co-occurrence (KEY OPTIMIZATION!)
        # This single operation replaces the triple-nested loop
        # Time complexity: O(M² × N) with BLAS optimization vs O(N × M²) with Python loops
        matrix_values = presence.values
        counts = matrix_values.T @ matrix_values

        # Create DataFrame with motif names as index and columns
        cooccurrence_df = pd.DataFrame(
            counts,
            index=motif_patterns,
            columns=motif_patterns
        )

        # Apply minimum threshold (except diagonal) - vectorized operation
        if min_cooccurrence > 1:
            n_motifs = len(motif_patterns)
            mask = np.eye(n_motifs, dtype=bool)
            cooccurrence_df.values[(cooccurrence_df.values < min_cooccurrence) & ~mask] = 0

        # Log statistics
        n_motifs = len(motif_patterns)
        total_cooccurrences = cooccurrence_df.values.sum() - np.diag(cooccurrence_df.values).sum()
        logging.info(f"  Total co-occurrences (above threshold): {int(total_cooccurrences/2)}")
        logging.info(f"  Motif pairs with co-occurrence >= {min_cooccurrence}: {np.sum(cooccurrence_df.values >= min_cooccurrence) - n_motifs}")

        return cooccurrence_df

    def calculate_motif_associations(self, assignments_df: pd.DataFrame,
                                    cooccurrence_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate association metrics between motif pairs.

        OPTIMIZED VERSION: Uses vectorized NumPy operations instead of nested loops.
        Expected speedup: 100-1000x faster for typical datasets.

        Args:
            assignments_df: DataFrame from assign_all_motifs_to_sequences()
            cooccurrence_df: Co-occurrence matrix from calculate_motif_cooccurrence_matrix()

        Returns:
            DataFrame with columns:
                - motif1: First motif core pattern
                - motif2: Second motif core pattern
                - shared_sequences: Number of sequences containing both
                - motif1_total: Total sequences containing motif1
                - motif2_total: Total sequences containing motif2
                - jaccard_similarity: |A∩B| / |A∪B|
                - conditional_prob_1_given_2: P(motif1|motif2)
                - conditional_prob_2_given_1: P(motif2|motif1)
        """
        logging.info(f"Calculating motif association metrics (optimized)...")

        motif_patterns = list(cooccurrence_df.index)

        # Get co-occurrence matrix as numpy array
        counts = cooccurrence_df.values
        totals = np.diag(counts)  # Diagonal contains total sequences per motif

        # Get upper triangle indices (avoid duplicates and self-pairs)
        n = len(motif_patterns)
        i_idx, j_idx = np.triu_indices(n, k=1)

        # Extract shared sequence counts for all pairs
        shared = counts[i_idx, j_idx]
        keep = shared > 0  # Only keep pairs with co-occurrence

        if not keep.any():
            # No associations found
            logging.info(f"  Found 0 motif pairs with co-occurrence")
            return pd.DataFrame(columns=[
                'motif1', 'motif2', 'shared_sequences',
                'motif1_total', 'motif2_total', 'jaccard_similarity',
                'conditional_prob_1_given_2', 'conditional_prob_2_given_1'
            ])

        # Filter to non-zero pairs only
        shared = shared[keep]
        sources = np.array(motif_patterns)[i_idx[keep]]
        targets = np.array(motif_patterns)[j_idx[keep]]
        total_i = totals[i_idx[keep]]
        total_j = totals[j_idx[keep]]

        # Vectorized metric calculations (ALL pairs computed in parallel!)
        union = total_i + total_j - shared

        # Use np.divide with out parameter to avoid division by zero warnings
        jaccard = np.divide(shared, union, out=np.zeros_like(shared, dtype=float), where=union != 0)
        cond_1_given_2 = np.divide(shared, total_j, out=np.zeros_like(shared, dtype=float), where=total_j != 0)
        cond_2_given_1 = np.divide(shared, total_i, out=np.zeros_like(shared, dtype=float), where=total_i != 0)

        # Create associations DataFrame
        associations_df = pd.DataFrame({
            'motif1': sources,
            'motif2': targets,
            'shared_sequences': shared.astype(int),
            'motif1_total': total_i.astype(int),
            'motif2_total': total_j.astype(int),
            'jaccard_similarity': jaccard,
            'conditional_prob_1_given_2': cond_1_given_2,
            'conditional_prob_2_given_1': cond_2_given_1
        })

        # Sort by shared sequences descending
        associations_df = associations_df.sort_values('shared_sequences', ascending=False)

        logging.info(f"  Found {len(associations_df)} motif pairs with co-occurrence")

        if len(associations_df) > 0:
            logging.info(f"  Average Jaccard similarity: {associations_df['jaccard_similarity'].mean():.3f}")
            logging.info(f"  Top association: {associations_df.iloc[0]['motif1']} <-> {associations_df.iloc[0]['motif2']} "
                        f"({associations_df.iloc[0]['shared_sequences']} shared sequences)")

        return associations_df

    def update_motif_counts_from_assignments(self, motifs: List[Dict],
                                            assignments_df: pd.DataFrame) -> List[Dict]:
        """
        Update motif counts and indices based on multi-label assignment scan.
        This updates the motif data to reflect ALL matching sequences, not just attention-based.

        Args:
            motifs: Original motifs from discovery
            assignments_df: DataFrame from assign_all_motifs_to_sequences()

        Returns:
            Updated motifs list with new 'scan_count', 'scan_indices' fields
        """
        logging.info(f"Updating motif counts from assignment scan...")

        for motif in motifs:
            core_pattern = motif.get('core_pattern', '')

            # Find all sequences that match this motif
            matching_sequences = assignments_df[
                assignments_df['motifs'].apply(lambda x: core_pattern in x)
            ]

            scan_indices = matching_sequences['sequence_index'].tolist()

            # Add scan-based counts alongside original attention-based counts
            motif['scan_count'] = len(scan_indices)
            motif['scan_indices'] = scan_indices

            # Calculate how many new sequences were found
            original_indices = set(motif.get('indices', []))
            new_indices = set(scan_indices) - original_indices

            motif['newly_discovered_count'] = len(new_indices)

        # Log summary
        total_attention = sum(m.get('count', 0) for m in motifs)
        total_scan = sum(m.get('scan_count', 0) for m in motifs)
        total_new = sum(m.get('newly_discovered_count', 0) for m in motifs)

        logging.info(f"  Attention-based assignments: {total_attention}")
        logging.info(f"  Scan-based assignments: {total_scan}")
        logging.info(f"  Newly discovered: {total_new} ({total_new/total_attention*100:.1f}% increase)")

        return motifs

    def create_cooccurrence_heatmap(self, cooccurrence_df: pd.DataFrame,
                                    output_path: Path,
                                    title: str = "Motif Co-occurrence Matrix"):
        """
        Create a heatmap visualization of the motif co-occurrence matrix.

        Args:
            cooccurrence_df: Co-occurrence matrix from calculate_motif_cooccurrence_matrix()
            output_path: Path to save the heatmap image
            title: Title for the plot
        """
        import seaborn as sns

        logging.info(f"Creating co-occurrence heatmap...")

        # Create figure
        n_motifs = len(cooccurrence_df)
        figsize = max(10, n_motifs * 0.6)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # Create heatmap
        sns.heatmap(
            cooccurrence_df,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Shared Sequences'},
            ax=ax
        )

        # Customize
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Motif', fontsize=12, fontweight='bold')
        ax.set_ylabel('Motif', fontsize=12, fontweight='bold')

        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"  Saved heatmap to {output_path}")

    def create_cooccurrence_network(self, cooccurrence_df: pd.DataFrame,
                                   motifs: List[Dict],
                                   output_path: Path,
                                   min_edge_weight: int = 2,
                                   max_edges_per_node: int = 10):
        """
        Create a network visualization where edges represent motif co-occurrence.
        Uses static layout (no physics) for fast rendering.

        Args:
            cooccurrence_df: Co-occurrence matrix from calculate_motif_cooccurrence_matrix()
            motifs: List of motifs (for node sizes)
            output_path: Path to save the network visualization
            min_edge_weight: Minimum shared sequences to draw an edge
            max_edges_per_node: Maximum edges to show per node (prevents overcrowding)
        """
        from pyvis.network import Network
        import math

        logging.info(f"Creating co-occurrence network...")

        # Create network with NO physics
        net = Network(height='900px', width='100%',
                     bgcolor='#222222', font_color='white',
                     directed=False)

        # Disable physics for fast rendering
        net.set_options("""
        {
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true,
                "keyboard": true
            }
        }
        """)

        # Create motif info dictionary
        motif_info = {}
        for motif in motifs:
            pattern = motif.get('core_pattern', '')
            motif_info[pattern] = {
                'count': motif.get('scan_count', motif.get('count', 0)),
                'original_count': motif.get('count', 0)
            }

        # Add nodes
        motif_patterns = list(cooccurrence_df.index)
        for pattern in motif_patterns:
            count = motif_info.get(pattern, {}).get('count', 0)

            # Node size based on sequence count (logarithmic scaling to handle wide ranges)
            node_size = 20 + (math.log10(count + 1) * 30)

            # Color based on sequence count
            if count > 50:
                color = '#ff6b6b'  # Red for high count
            elif count > 20:
                color = '#ffa500'  # Orange for medium count
            else:
                color = '#4dabf7'  # Blue for low count

            net.add_node(
                pattern,
                label=pattern,
                size=node_size,
                color=color,
                title=f"{pattern}<br>Sequences: {count}"
            )

        # Aggressive edge filtering: only keep top edges per node
        logging.info(f"  Filtering edges: min_weight={min_edge_weight}, max_per_node={max_edges_per_node}")

        # Collect all edges with their weights
        all_edges = []
        for i, motif1 in enumerate(motif_patterns):
            for j, motif2 in enumerate(motif_patterns):
                if i >= j:  # Only upper triangle
                    continue

                shared = cooccurrence_df.loc[motif1, motif2]

                if shared >= min_edge_weight:
                    all_edges.append((motif1, motif2, shared))

        # For each node, keep only top N edges
        node_edges = {pattern: [] for pattern in motif_patterns}
        for m1, m2, weight in all_edges:
            node_edges[m1].append((m2, weight))
            node_edges[m2].append((m1, weight))

        # Filter to top edges per node
        edges_to_add = set()
        for node, edges in node_edges.items():
            # Sort by weight descending and take top N
            top_edges = sorted(edges, key=lambda x: x[1], reverse=True)[:max_edges_per_node]
            for neighbor, weight in top_edges:
                # Add as tuple with smaller pattern first (to avoid duplicates)
                edge = tuple(sorted([node, neighbor]))
                edges_to_add.add(edge + (weight,))

        # Add filtered edges
        edge_count = 0
        for motif1, motif2, shared in edges_to_add:
            # Edge width based on shared sequences (logarithmic scaling)
            edge_width = 1 + (math.log10(shared + 1) * 2)

            # Edge color intensity based on shared sequences
            alpha = min(1.0, shared / 20)

            net.add_edge(
                motif1,
                motif2,
                value=float(shared),
                width=edge_width,
                title=f"Shared sequences: {shared}",
                color=f'rgba(100, 200, 255, {alpha})'
            )
            edge_count += 1

        # Save
        net.save_graph(str(output_path))

        logging.info(f"  Added {len(motif_patterns)} nodes and {edge_count} edges (filtered from {len(all_edges)})")
        logging.info(f"  Saved co-occurrence network to {output_path}")

    def _create_logo_plots_comparison_overview(self, motifs: List[Dict], output_dir: Path):
        """Create overview showing logo plots for multiple motifs (most common combination each)"""

        fig, axes = plt.subplots(len(motifs), 1, figsize=(16, len(motifs) * 3))
        if len(motifs) == 1:
            axes = [axes]

        amino_acids = list("ARNDCQEGHILKMFPSTWYV")
        colors = plt.cm.Set3(np.linspace(0, 1, len(amino_acids)))

        for motif_idx, motif in enumerate(motifs):
            core_pattern = motif.get('core_pattern', f'Motif_{motif_idx+1}')
            most_common_combo = motif.get('most_common_combination', '')
            length_combinations = motif.get('length_combinations', {})

            if not most_common_combo or most_common_combo not in length_combinations:
                continue

            # Use the most common length combination for this overview
            pssm_data = length_combinations[most_common_combo]
            pssm_matrix = pssm_data['pssm_matrix']
            consensus = pssm_data['consensus_pattern']
            count = pssm_data['sequence_count']

            ax = axes[motif_idx]
            seq_length = len(pssm_matrix)

            # Create logo plot for this motif's most common combination
            for pos_idx, pos_data in enumerate(pssm_matrix):
                aa_freqs = pos_data.get('amino_acid_frequencies', {})
                info_content = pos_data.get('information_content', 0)

                sorted_aas = sorted(aa_freqs.items(), key=lambda x: x[1], reverse=True)

                y_offset = 0
                for aa, freq in sorted_aas:
                    if freq > 0.05:  # Show AAs with >5% frequency
                        height = freq * info_content
                        aa_idx = amino_acids.index(aa)

                        rect = plt.Rectangle((pos_idx - 0.4, y_offset), 0.8, height,
                                           facecolor=colors[aa_idx], alpha=0.8,
                                           edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)

                        if height > 0.15:
                            ax.text(pos_idx, y_offset + height/2, aa,
                                   ha='center', va='center', fontweight='bold',
                                   fontsize=11, color='white')

                        y_offset += height

            ax.set_xlim(-0.5, seq_length - 0.5)
            max_height = max(2.0, max([pos_data.get('information_content', 0) for pos_data in pssm_matrix]) * 1.1)
            ax.set_ylim(0, max_height)
            ax.set_xlabel('Position')
            ax.set_ylabel('Information (bits)')
            ax.set_title(f'{core_pattern} - {most_common_combo} ({count} sequences)')
            ax.set_xticks(range(seq_length))
            ax.grid(True, alpha=0.3)

        plt.suptitle('Sequence Logo Comparison: Most Common Length Combinations', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'visualizations' / 'motifs_logo_comparison_overview.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logging.info("Created logo plot comparison overview")


### CONSOLIDATION FUNCTIONS ###

def parse_motif(motif):
    """
    Parse a motif to extract its components.
    Returns the base pattern with brackets and the surrounding amino acids.
    """
    # Find the bracketed pattern
    bracket_match = re.search(r'([A-Z]*)\[(\d+)\]([A-Z]*)', motif)
    if bracket_match:
        before = bracket_match.group(1)
        number = bracket_match.group(2)
        after = bracket_match.group(3)
        core = f"{before[-1] if before else ''}[{number}]{after[0] if after else ''}"
        prefix = before[:-1] if len(before) > 1 else ""
        suffix = after[1:] if len(after) > 1 else ""
        return {
            'core': core,
            'prefix': prefix,
            'suffix': suffix,
            'full': motif,
            'number': int(number)
        }
    return None

def is_parent_of_old(parent_motif, child_motif):
    """
    OLD VERSION: Check if parent_motif is a parent of child_motif.
    A parent is a substring that maintains the bracket structure.

    NOTE: This is the old implementation kept for backward compatibility.
    New code should use is_parent_of() from line 72 instead.
    """
    parent_parsed = parse_motif(parent_motif)
    child_parsed = parse_motif(child_motif)

    if not parent_parsed or not child_parsed:
        return False

    # Must have the same bracket number
    if parent_parsed['number'] != child_parsed['number']:
        return False

    # Check if parent is contained in child
    # This means child has all the components of parent plus potentially more
    if parent_motif in child_motif:
        # Additional check: make sure the bracket structure is preserved
        if f"[{parent_parsed['number']}]" in child_motif:
            return True

    return False

def find_motif_hierarchy(motifs_df):
    """
    Build a hierarchy of motifs based on parent-child relationships.
    Uses old is_parent_of_old() logic for backward compatibility.
    """
    motifs = motifs_df['core_pattern'].tolist()
    hierarchy = defaultdict(list)
    parent_map = {}

    # Sort motifs by length (shorter ones are more likely to be parents)
    motifs_sorted = sorted(motifs, key=len)

    for i, potential_parent in enumerate(motifs_sorted):
        for j, potential_child in enumerate(motifs_sorted):
            if i != j and is_parent_of_old(potential_parent, potential_child):
                hierarchy[potential_parent].append(potential_child)
                # Keep track of the most immediate parent
                if potential_child not in parent_map or len(potential_parent) > len(parent_map[potential_child]):
                    parent_map[potential_child] = potential_parent

    return hierarchy, parent_map

def consolidate_motif_dictionaries(motifs: List[Dict], hierarchy: Dict) -> List[Dict]:
    """
    Consolidate motif dictionaries by merging child motifs into parents.
    This combines sequences and indices from children into parents.
    """
    import logging
    from collections import defaultdict

    # Create a mapping from pattern to motif
    pattern_to_motif = {motif.get('core_pattern', motif.get('pattern', '')): motif for motif in motifs}

    # Track which motifs have been merged into parents
    merged_children = set()

    # Helper function to get all descendants
    def get_all_descendants(motif_pattern, hierarchy, visited=None):
        if visited is None:
            visited = set()
        if motif_pattern in visited:
            return []
        visited.add(motif_pattern)

        descendants = []
        if motif_pattern in hierarchy:
            for child in hierarchy[motif_pattern]:
                descendants.append(child)
                descendants.extend(get_all_descendants(child, hierarchy, visited))
        return descendants

    # For each parent motif, consolidate all children
    for parent_pattern in hierarchy:
        if parent_pattern not in pattern_to_motif:
            continue

        parent_motif = pattern_to_motif[parent_pattern]
        all_descendants = get_all_descendants(parent_pattern, hierarchy)

        # Remove duplicates
        unique_descendants = []
        seen = set()
        for desc in all_descendants:
            if desc not in seen and desc in pattern_to_motif:
                seen.add(desc)
                unique_descendants.append(desc)

        # Merge all descendant sequences and indices into parent
        for child_pattern in unique_descendants:
            if child_pattern in pattern_to_motif:
                child_motif = pattern_to_motif[child_pattern]

                # Merge sequences and indices
                parent_motif['sequences'].extend(child_motif.get('sequences', []))
                parent_motif['indices'].extend(child_motif.get('indices', []))
                parent_motif['count'] = len(parent_motif['indices'])

                # Mark child as merged
                merged_children.add(child_pattern)

        # Store metadata about consolidation
        parent_motif['child_motifs'] = unique_descendants[:10]  # First 10
        parent_motif['num_children'] = len(unique_descendants)
        parent_motif['original_count'] = parent_motif.get('original_count', parent_motif['count'] - sum(
            pattern_to_motif[c]['count'] for c in unique_descendants if c in pattern_to_motif
        ))

        # Mark parent as needing PSSM rebuild since sequences changed
        parent_motif['needs_pssm_rebuild'] = True
        parent_motif['length_combinations'] = {}  # Clear old PSSMs

    # Return only parent motifs (exclude merged children)
    consolidated_motifs = [
        motif for motif in motifs
        if motif.get('core_pattern', motif.get('pattern', '')) not in merged_children
    ]

    logging.info(f"Consolidated {len(motifs)} motifs into {len(consolidated_motifs)} (merged {len(merged_children)} children)")

    return consolidated_motifs

def export_sequences_for_motifs(motifs: List[Dict], sequences: List[str], output_dir: Path):
    """Export all sequences for each consolidated motif to separate files"""
    seq_export_dir = output_dir / 'motif_sequences'
    seq_export_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Exporting sequences for {len(motifs)} motifs to {seq_export_dir}")

    for i, motif in enumerate(motifs):
        core_pattern = motif.get('core_pattern', f'motif_{i+1}')
        # Sanitize filename
        safe_pattern = core_pattern.replace('[', '_').replace(']', '_').replace('/', '_')

        # Export all sequences
        seq_file = seq_export_dir / f'{safe_pattern}_sequences.txt'
        with open(seq_file, 'w') as f:
            f.write(f"# Motif: {core_pattern}\n")
            f.write(f"# Consensus: {motif.get('consensus_pattern', 'N/A')}\n")
            f.write(f"# Total sequences: {motif['count']}\n")
            f.write(f"# Length combinations: {motif.get('num_length_combinations', 0)}\n")
            f.write("#\n")

            motif_sequences = [sequences[idx] for idx in motif['indices']]
            for idx, seq in zip(motif['indices'], motif_sequences):
                f.write(f"{idx}\t{seq}\n")

        # Also export length-stratified sequences if available
        if motif.get('length_combinations'):
            for combo_key, combo_data in motif['length_combinations'].items():
                combo_file = seq_export_dir / f'{safe_pattern}_{combo_key}_sequences.txt'
                with open(combo_file, 'w') as f:
                    f.write(f"# Motif: {core_pattern} - Length combo: {combo_key}\n")
                    f.write(f"# Consensus: {combo_data.get('consensus_pattern', 'N/A')}\n")
                    f.write(f"# Sequences: {combo_data['sequence_count']}\n")
                    f.write("#\n")
                    for idx, seq in zip(combo_data['indices'], combo_data['sequences']):
                        f.write(f"{idx}\t{seq}\n")

    logging.info(f"Exported sequences to {len(list(seq_export_dir.glob('*.txt')))} files")

def export_sequences_with_motifs_csv(motifs: List[Dict], sequences: List[str], output_dir: Path,
                                      method_name: str = None, percentile: int = None,
                                      consolidated: bool = False):
    """
    Export all sequences that have detected motifs to a CSV file.
    Each sequence appears once with all its motifs (parent and child) separated by semicolons.

    Args:
        motifs: List of motif dictionaries (before filtering out children)
        sequences: List of all sequences
        output_dir: Directory to save the CSV file
        method_name: Optional method name (e.g., 'seq_relative') to include in filename
        percentile: Optional percentile value (e.g., 70) to include in filename
        consolidated: Whether the motifs have been consolidated (affects filename)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reverse mapping: sequence -> list of motif patterns
    sequence_to_motifs = defaultdict(set)

    for motif in motifs:
        core_pattern = motif.get('core_pattern', motif.get('pattern', 'Unknown'))
        motif_sequences = motif.get('sequences', [])

        # Add this motif pattern to all sequences that contain it
        for seq in motif_sequences:
            sequence_to_motifs[seq].add(core_pattern)

    # Build descriptive filename with method/percentile info
    if method_name and percentile is not None:
        filename = f'sequences_with_motifs_{method_name}_p{percentile}'
        if consolidated:
            filename += '_consolidated'
        filename += '.csv'
    else:
        filename = 'sequences_with_motifs.csv'

    # Export to CSV
    csv_file = output_dir / filename

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_index', 'sequence', 'motifs'])

        # Iterate through sequences and write those with motifs
        for idx, seq in enumerate(sequences):
            if seq in sequence_to_motifs:
                # Sort motifs for consistent output
                motif_list = sorted(sequence_to_motifs[seq])
                motifs_str = ';'.join(motif_list)
                writer.writerow([idx, seq, motifs_str])

    num_sequences_with_motifs = len(sequence_to_motifs)
    logging.info(f"Exported {num_sequences_with_motifs} sequences with motifs to {csv_file}")

def create_interactive_network_pyvis(sequences_csv_path, save_path='motif_network_interactive.html'):
    """
    Create an interactive network visualization using PyVis.
    Uses static hierarchical layout (no physics) for fast rendering.

    Args:
        sequences_csv_path: Path to CSV with columns: sequence_index, sequence, motifs
        save_path: Where to save the HTML file
    """
    # Load sequences CSV and extract all unique motifs
    df = pd.read_csv(sequences_csv_path)

    # Parse motifs from semicolon-separated list
    all_motifs = set()
    motif_to_sequences = {}

    for _, row in df.iterrows():
        if pd.isna(row['motifs']):
            continue
        motifs_list = [m.strip() for m in str(row['motifs']).split(';')]
        for motif in motifs_list:
            all_motifs.add(motif)
            if motif not in motif_to_sequences:
                motif_to_sequences[motif] = []
            motif_to_sequences[motif].append(row['sequence'])

    # Reconstruct hierarchy from motif patterns
    motifs_list = sorted(all_motifs)
    hierarchy, parent_map = build_hierarchy_from_motifs(motifs_list)

    logging.info(f"Loaded {len(all_motifs)} unique motifs from sequences CSV")
    logging.info(f"Reconstructed hierarchy with {len(hierarchy)} parent motifs")

    # Create network
    net = Network(height='900px', width='100%',
                  bgcolor='#222222', font_color='white',
                  directed=True)

    # Configure Barnes-Hut physics for proper node spreading (matches webapp)
    # Must be in set_options to avoid being overwritten
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0
            },
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        },
        "configure": {
            "enabled": true,
            "filter": ["physics"]
        }
    }
    """)

    # Create motif info dictionary for ALL motifs (parents + children)
    motif_info = {}
    for motif in all_motifs:
        count = len(motif_to_sequences.get(motif, []))
        num_children = len(hierarchy.get(motif, []))
        motif_info[motif] = {
            'count': count,
            'children': num_children
        }

    # Find root nodes
    all_children = set()
    for children in hierarchy.values():
        all_children.update(children)
    root_nodes = set(motif_info.keys()) - all_children

    # Calculate hierarchy levels (fix mutable default argument)
    def get_level(node, hierarchy, level_cache=None):
        if level_cache is None:
            level_cache = {}

        if node in level_cache:
            return level_cache[node]

        if node in root_nodes:
            level = 0
        else:
            # Find parent
            parent = None
            for p, children in hierarchy.items():
                if node in children:
                    parent = p
                    break
            if parent:
                level = get_level(parent, hierarchy, level_cache) + 1
            else:
                level = 0

        level_cache[node] = level
        return level

    # Build level cache once
    level_cache = {}
    for motif in motif_info.keys():
        get_level(motif, hierarchy, level_cache)

    # Add nodes for ALL motifs
    for motif, info in motif_info.items():
        count = info['count']
        children_count = info['children']

        # Size based on count (log scale)
        size = max(10, min(50, np.log1p(count) * 5))

        # Color based on hierarchy level
        level = level_cache.get(motif, 0)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        color = colors[level % len(colors)]

        # Create hover text
        title = (f"<b>{motif}</b><br>"
                f"Sequences: {count:,}<br>"
                f"Children: {children_count}<br>"
                f"Level: {level}")

        net.add_node(motif,
                    label=motif,
                    title=title,
                    size=size,
                    color=color,
                    level=level,
                    font={'size': max(8, min(14, size/3))})

    # Add edges (all parent-child relationships from hierarchy)
    edge_count = 0
    for parent, children in hierarchy.items():
        for child in children:
            # Both parent and child should exist now since we include all motifs
            net.add_edge(parent, child, arrows='to', color='#666666', width=2)
            edge_count += 1

    # Save network - convert Path to string
    net.save_graph(str(save_path))
    print(f"[OK] Interactive parent-child network saved to {save_path}")
    print(f"     Added {len(motif_info)} nodes and {edge_count} edges")
    logging.info(f"Created network with {len(motif_info)} nodes and {edge_count} edges")
    return net

def create_interactive_plotly_tree(sequences_csv_path, save_path='motif_tree_interactive.html'):
    """
    Create an interactive tree/sunburst visualization using Plotly.

    Args:
        sequences_csv_path: Path to CSV with columns: sequence_index, sequence, motifs
        save_path: Where to save the HTML file
    """
    # Load sequences CSV and extract all unique motifs
    df = pd.read_csv(sequences_csv_path)

    # Parse motifs from semicolon-separated list
    all_motifs = set()
    motif_to_sequences = {}

    for _, row in df.iterrows():
        if pd.isna(row['motifs']):
            continue
        motifs_list = [m.strip() for m in str(row['motifs']).split(';')]
        for motif in motifs_list:
            all_motifs.add(motif)
            if motif not in motif_to_sequences:
                motif_to_sequences[motif] = []
            motif_to_sequences[motif].append(row['sequence'])

    # Reconstruct hierarchy from motif patterns
    motifs_list = sorted(all_motifs)
    hierarchy, parent_map = build_hierarchy_from_motifs(motifs_list)

    # Prepare data for treemap/sunburst
    labels = []
    parents = []
    values = []
    colors = []
    hover_text = []

    # Create motif info dictionary for ALL motifs (parents + children)
    motif_info = {}
    for motif in all_motifs:
        count = len(motif_to_sequences.get(motif, []))
        num_children = len(hierarchy.get(motif, []))
        motif_info[motif] = {
            'count': count,
            'children': num_children
        }

    # Find root nodes
    all_children = set()
    for children_list in hierarchy.values():
        all_children.update(children_list)
    root_nodes = set(motif_info.keys()) - all_children

    # Calculate total sequences for root node (sum of all root-level motifs)
    # This is required for Plotly sunburst to render properly
    total_sequences = sum(info['count'] for motif, info in motif_info.items() if motif in root_nodes)

    # Add root node with proper value
    labels.append("All Motifs")
    parents.append("")
    values.append(total_sequences)  # Root must have sum of root children
    colors.append(0)
    hover_text.append(f"Root<br>Total: {total_sequences:,}")

    # Add all motifs (including children!)
    for motif, info in motif_info.items():
        labels.append(motif)

        # Find parent - use parent_map from hierarchy reconstruction
        parent_motif = parent_map.get(motif)

        if parent_motif:
            parents.append(parent_motif)
        else:
            # Root motifs go under "All Motifs"
            parents.append("All Motifs")

        values.append(info['count'])  # Each motif gets its own count
        colors.append(info['count'])
        hover_text.append(f"Sequences: {info['count']:,}<br>Children: {info['children']}")

    # Create sunburst chart
    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        text=labels,
        hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
        customdata=hover_text,
        marker=dict(
            colors=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sequence Count")
        )
    ))

    fig_sunburst.update_layout(
        title="Motif Hierarchy - Interactive Sunburst (Click to zoom)",
        width=1000,
        height=1000,
        template='plotly_dark'
    )

    # Create treemap
    fig_treemap = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=labels,
        hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
        customdata=hover_text,
        marker=dict(
            colors=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sequence Count")
        ),
        textposition="middle center",
        pathbar=dict(visible=True)
    ))

    fig_treemap.update_layout(
        title="Motif Hierarchy - Interactive Treemap (Click to drill down)",
        width=1200,
        height=800,
        template='plotly_dark'
    )

    # Save both visualizations
    fig_sunburst.write_html(str(save_path).replace('.html', '_sunburst.html'))
    fig_treemap.write_html(str(save_path).replace('.html', '_treemap.html'))

    print(f"[OK] Interactive sunburst saved to {str(save_path).replace('.html', '_sunburst.html')}")
    print(f"[OK] Interactive treemap saved to {str(save_path).replace('.html', '_treemap.html')}")
    logging.info(f"Created tree visualizations with {len(motif_info)} motifs")

    return fig_sunburst, fig_treemap

def create_dashboard(consolidated_df, hierarchy, save_path='motif_dashboard.html'):
    """
    Create a comprehensive interactive dashboard.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 20 Motifs by Consolidated Count',
                       'Count Increase Distribution',
                       'Children Distribution',
                       'Motif Length vs Count'),
        specs=[[{'type': 'bar'}, {'type': 'histogram'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    # Sort by consolidated count
    df_sorted = consolidated_df.sort_values('total_sequences', ascending=False)

    # 1. Top motifs
    top_20 = df_sorted.head(20)
    fig.add_trace(
        go.Bar(x=top_20['core_pattern'],
               y=top_20['total_sequences'],
               name='Consolidated',
               marker_color='lightblue',
               hovertemplate='%{x}<br>Count: %{y:,}<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=top_20['core_pattern'],
               y=top_20['original_count'],
               name='Original',
               marker_color='coral',
               hovertemplate='%{x}<br>Count: %{y:,}<extra></extra>'),
        row=1, col=1
    )

    # 2. Count increase distribution
    count_increase_pct = ((df_sorted['total_sequences'] - df_sorted['original_count']) /
                         df_sorted['original_count'].replace(0, 1) * 100)
    count_increase_pct = count_increase_pct[count_increase_pct > 0]

    fig.add_trace(
        go.Histogram(x=count_increase_pct,
                    nbinsx=30,
                    marker_color='green',
                    name='Count Increase %',
                    hovertemplate='Increase: %{x:.1f}%<br>Count: %{y}<extra></extra>'),
        row=1, col=2
    )

    # 3. Children distribution
    children_counts = df_sorted['num_children'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=children_counts.index,
               y=children_counts.values,
               marker_color='purple',
               name='# Motifs',
               hovertemplate='Children: %{x}<br>Motifs: %{y}<extra></extra>'),
        row=2, col=1
    )

    # 4. Motif length vs count
    df_sorted['motif_length'] = df_sorted['core_pattern'].str.len()
    fig.add_trace(
        go.Scatter(x=df_sorted['motif_length'],
                  y=df_sorted['total_sequences'],
                  mode='markers',
                  marker=dict(
                      size=np.log1p(df_sorted['num_children']) * 3 + 3,
                      color=df_sorted['num_children'],
                      colorscale='Viridis',
                      showscale=True,
                      colorbar=dict(title="Children", x=1.15)
                  ),
                  text=df_sorted['core_pattern'],
                  name='Motifs',
                  hovertemplate='%{text}<br>Length: %{x}<br>Count: %{y:,}<extra></extra>'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Motif Consolidation Dashboard",
        showlegend=True,
        template='plotly_dark',
        height=800,
        width=1400
    )

    # Update axes
    fig.update_xaxes(title_text="Motif", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Count Increase (%)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Children", row=2, col=1)
    fig.update_xaxes(title_text="Motif Length", row=2, col=2)

    fig.update_yaxes(title_text="Count", row=1, col=1, type='log')
    fig.update_yaxes(title_text="Number of Motifs", row=1, col=2)
    fig.update_yaxes(title_text="Number of Motifs", row=2, col=1)
    fig.update_yaxes(title_text="Consolidated Count", row=2, col=2, type='log')

    fig.write_html(str(save_path))
    print(f"Interactive dashboard saved to {save_path}")
    return fig

def create_static_summary(original_df, consolidated_df, save_path='motif_summary.png'):
    """
    Create static summary statistics visualization (kept from original).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Top motifs comparison
    ax = axes[0, 0]
    top_n = 15
    top_consolidated = consolidated_df.nlargest(top_n, 'total_sequences')

    y_pos = np.arange(len(top_consolidated))
    ax.barh(y_pos, top_consolidated['total_sequences'], alpha=0.7, label='Consolidated')
    ax.barh(y_pos, top_consolidated['original_count'], alpha=0.7, label='Original')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_consolidated['core_pattern'])
    ax.set_xlabel('Total Sequences')
    ax.set_title(f'Top {top_n} Motifs - Original vs Consolidated Counts')
    ax.legend()
    ax.invert_yaxis()

    # 2. Distribution of child counts
    ax = axes[0, 1]
    child_counts = consolidated_df['num_children'].value_counts().sort_index()
    ax.bar(child_counts.index, child_counts.values, alpha=0.7, color='green')
    ax.set_xlabel('Number of Children')
    ax.set_ylabel('Number of Parent Motifs')
    ax.set_title('Distribution of Child Motifs per Parent')
    ax.grid(True, alpha=0.3)

    # 3. Count increase distribution
    ax = axes[1, 0]
    count_increase = consolidated_df['total_sequences'] - consolidated_df['original_count']
    count_increase_pct = (count_increase / consolidated_df['original_count'].replace(0, 1)) * 100

    ax.hist(count_increase_pct[count_increase_pct > 0], bins=30, alpha=0.7, color='orange')
    ax.set_xlabel('Count Increase (%)')
    ax.set_ylabel('Number of Motifs')
    ax.set_title('Distribution of Count Increases from Consolidation')
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    summary_data = [
        ['Metric', 'Original', 'Consolidated'],
        ['Total Motifs', len(original_df), len(consolidated_df)],
        ['Total Sequences (sum)', f"{original_df['total_sequences'].sum():,}",
         f"{consolidated_df['total_sequences'].sum():,}"],
        ['Max Count', f"{original_df['total_sequences'].max():,}",
         f"{consolidated_df['total_sequences'].max():,}"],
        ['Mean Count', f"{original_df['total_sequences'].mean():.1f}",
         f"{consolidated_df['total_sequences'].mean():.1f}"],
        ['Motifs with Children', '-',
         f"{(consolidated_df['num_children'] > 0).sum()}"],
        ['Max Children per Motif', '-',
         f"{consolidated_df['num_children'].max()}"]
    ]

    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Motif Consolidation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    print(f"Static summary saved to {save_path}")
    plt.close()


### MAIN EXECUTION ###

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Position-Independent Motif Discovery for Protein Sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--csv', help='Path to CSV file')
    data_group.add_argument('--test', action='store_true', help='Use built-in test dataset')

    parser.add_argument('--column', default='sequence', help='Column name for sequences')
    parser.add_argument('--models', type=int, default=2, help='Number of models to train')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--min-instances', type=int, default=30, help='Minimum sequences per motif')
    parser.add_argument('--no-structural-masking', action='store_true', help='Disable variable-only masking')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--output-dir', default='motif_analysis_results', help='Output directory')

    # Validation arguments
    parser.add_argument('--enable-validation', action='store_true',
                       help='Enable validation to determine optimal epochs (splits data 80/20)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Number of epochs without improvement before stopping (default: 3)')

    return parser.parse_args()

def main():
    # Parse simplified arguments
    args = parse_arguments()

    # Setup logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if args.log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)

    logging.info("="*60)
    logging.info("POSITION-INDEPENDENT MOTIF DISCOVERY")
    logging.info("="*60)

    # Initialize discovery
    discovery = SelfAttentionMotifDiscovery(num_models=args.models)

    # Configure masking
    if args.no_structural_masking:
        discovery.mask_structural_positions = False
        logging.info("Variable-only masking: DISABLED")
    else:
        discovery.mask_structural_positions = True
        logging.info("Variable-only masking: ENABLED (never masks A/C positions)")

    # Load data
    if args.test:
        logging.info("Using test dataset")
        sequences, true_labels = discovery.create_test_dataset()
    else:
        logging.info(f"Loading sequences from: {args.csv}")
        sequences = discovery.load_sequences_from_csv(args.csv, args.column)

    # Create output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Train models (with optional validation)
    discovery.train_models(
        sequences,
        epochs=args.epochs,
        enable_validation=args.enable_validation,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=output_base_dir
    )

    # Extract attention
    attention_vectors = discovery.extract_attention_weights(sequences)

    # Run position-independent discovery for each percentile
    percentiles = [65, 70, 75]

    for percentile in percentiles:
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING PERCENTILE {percentile}")
        logging.info(f"{'='*60}")

        # Create output directory for this percentile
        output_dir = Path(args.output_dir) / f'p{percentile}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover motifs
        motifs = discovery._discover_motifs_position_independent(
            sequences, attention_vectors, percentile, args.min_instances
        )

        if not motifs:
            logging.warning(f"No motifs found for p{percentile}")
            continue

        # Consolidate motifs (always enabled)
        logging.info(f"  Consolidating motifs...")
        df_temp = pd.DataFrame([{
            'core_pattern': m.get('core_pattern', m.get('pattern', '')),
            'total_sequences': m.get('count', 0)
        } for m in motifs])

        hierarchy, parent_map = find_motif_hierarchy(df_temp)
        motifs = consolidate_motif_dictionaries(motifs, hierarchy)

        # Export sequences
        export_sequences_for_motifs(motifs, sequences, output_dir)
        export_sequences_with_motifs_csv(motifs, sequences, output_dir,
                                          method_name='position_independent',
                                          percentile=percentile,
                                          consolidated=True)

        # Build length-stratified PSSMs from motif assignments
        logging.info(f"  Building length-stratified PSSMs from assignments...")
        motifs = discovery.build_pssms_from_assignments(motifs, sequences, output_dir)

        # Perform multi-label assignment (scan all sequences for all motifs)
        logging.info(f"  Performing multi-label motif assignment...")
        assignments_df = discovery.assign_all_motifs_to_sequences(motifs, sequences)

        # Calculate co-occurrence matrix and associations
        logging.info(f"  Calculating motif co-occurrence...")
        cooccurrence_df = discovery.calculate_motif_cooccurrence_matrix(
            assignments_df, motifs, min_cooccurrence=2
        )
        associations_df = discovery.calculate_motif_associations(assignments_df, cooccurrence_df)

        # Update motif counts with scan-based assignments
        motifs = discovery.update_motif_counts_from_assignments(motifs, assignments_df)

        # Create co-occurrence visualizations
        logging.info(f"  Creating co-occurrence visualizations...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Heatmap
        heatmap_path = viz_dir / f'cooccurrence_heatmap_p{percentile}.png'
        discovery.create_cooccurrence_heatmap(cooccurrence_df, heatmap_path)

        # Co-occurrence network
        cooccurrence_network_path = output_dir / f'motif_cooccurrence_network_p{percentile}.html'
        discovery.create_cooccurrence_network(cooccurrence_df, motifs, cooccurrence_network_path)

        # Export multi-label assignments to CSV
        assignments_csv = output_dir / f'sequence_assignments_multilabel_p{percentile}.csv'
        # Convert list column to string for CSV export
        assignments_export = assignments_df.copy()
        assignments_export['motifs'] = assignments_export['motifs'].apply(lambda x: '; '.join(x) if x else '')
        assignments_export.to_csv(assignments_csv, index=False)
        logging.info(f"  Saved multi-label assignments to {assignments_csv}")

        # Export associations to CSV
        if len(associations_df) > 0:
            associations_csv = output_dir / f'motif_associations_p{percentile}.csv'
            associations_df.to_csv(associations_csv, index=False)
            logging.info(f"  Saved motif associations to {associations_csv}")

        # Save motifs CSV (with updated scan counts)
        csv_file = output_dir / f'motifs_p{percentile}_consolidated.csv'
        discovery._save_position_independent_motifs(motifs, csv_file)

        # Create PSSM visualizations
        logging.info(f"  Creating PSSM visualizations...")
        pssm_output_dir = output_dir / 'visualizations'
        pssm_output_dir.mkdir(parents=True, exist_ok=True)
        discovery.create_pssm_heatmaps_UPDATED(motifs, attention_vectors, output_dir, top_n=10)

        # Create hierarchy visualizations (using sequence CSV for complete hierarchy)
        logging.info(f"  Creating hierarchy visualizations...")
        sequences_csv = output_dir / f'sequences_with_motifs_position_independent_p{percentile}_consolidated.csv'

        create_interactive_network_pyvis(sequences_csv,
            output_dir / 'motif_network_interactive.html')
        create_interactive_plotly_tree(sequences_csv,
            output_dir / 'motif_tree_interactive.html')

        # Create dashboard and summary (still using motif summary CSV for consolidated view)
        motif_df = pd.read_csv(csv_file)
        hierarchy_for_dashboard, parent_map_for_dashboard = find_motif_hierarchy(motif_df)

        create_dashboard(motif_df, hierarchy_for_dashboard,
            output_dir / 'motif_dashboard.html')
        create_static_summary(motif_df, motif_df,
            output_dir / 'motif_summary.png')

        logging.info(f"✓ Percentile {percentile} complete: {output_dir}")

    logging.info("\n" + "="*60)
    logging.info("ALL PERCENTILES COMPLETE!")
    logging.info("="*60)

if __name__ == "__main__":
    main()
