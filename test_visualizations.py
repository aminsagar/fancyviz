#!/usr/bin/env python3
"""
Standalone Test Script for Motif Hierarchy Visualizations
==========================================================

This script allows quick testing of visualization functions without running
the full motif discovery pipeline.

Usage:
    python test_visualizations.py <path_to_sequences_csv>

Example:
    python test_visualizations.py results/p65/sequences_with_motifs_position_independent_p65_consolidated.csv

Outputs:
    - test_output/motif_network_interactive.html (PyVis network)
    - test_output/motif_tree_interactive_sunburst.html (Plotly sunburst)
    - test_output/motif_tree_interactive_treemap.html (Plotly treemap)
"""

import argparse
import sys
from pathlib import Path
import logging

# Import visualization functions from the main script
from motif_discovery_position_independent import (
    create_interactive_network_pyvis,
    create_interactive_plotly_tree
)


def setup_logging():
    """Configure logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_csv(csv_path: Path) -> bool:
    """
    Validate that the CSV file exists and has the required columns.

    Args:
        csv_path: Path to the sequences CSV file

    Returns:
        True if valid, False otherwise
    """
    if not csv_path.exists():
        logging.error(f"CSV file not found: {csv_path}")
        return False

    # Quick check: read first line to verify columns
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        required_columns = ['sequence_index', 'sequence', 'motifs']

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logging.error(f"CSV missing required columns: {missing_columns}")
            logging.error(f"Found columns: {df.columns.tolist()}")
            return False

        logging.info(f"✅ CSV file validated: {csv_path}")
        logging.info(f"   Columns: {df.columns.tolist()}")
        return True

    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return False


def test_visualizations(csv_path: Path, output_dir: Path):
    """
    Test the visualization functions with the provided CSV.

    Args:
        csv_path: Path to sequences CSV
        output_dir: Directory to save output files
    """
    logging.info("="*60)
    logging.info("TESTING MOTIF VISUALIZATIONS")
    logging.info("="*60)
    logging.info(f"Input CSV: {csv_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test 1: Network visualization
        logging.info("📊 Creating interactive network visualization...")
        network_path = output_dir / 'motif_network_interactive.html'
        create_interactive_network_pyvis(csv_path, network_path)
        logging.info(f"   Saved to: {network_path}")
        logging.info("")

        # Test 2: Tree visualizations (sunburst + treemap)
        logging.info("📊 Creating interactive tree visualizations...")
        tree_path = output_dir / 'motif_tree_interactive.html'
        create_interactive_plotly_tree(csv_path, tree_path)
        logging.info(f"   Saved to: {tree_path.parent / (tree_path.stem + '_sunburst.html')}")
        logging.info(f"   Saved to: {tree_path.parent / (tree_path.stem + '_treemap.html')}")
        logging.info("")

        logging.info("="*60)
        logging.info("✅ ALL TESTS PASSED!")
        logging.info("="*60)
        logging.info("\nGenerated files:")
        logging.info(f"  1. {network_path.name}")
        logging.info(f"  2. {tree_path.stem}_sunburst.html")
        logging.info(f"  3. {tree_path.stem}_treemap.html")
        logging.info("\nOpen these HTML files in your browser to view the visualizations.")

        return True

    except Exception as e:
        logging.error("="*60)
        logging.error("❌ TEST FAILED!")
        logging.error("="*60)
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for the test script."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Test motif hierarchy visualizations with a sequences CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a specific percentile result
  python test_visualizations.py results/p65/sequences_with_motifs_position_independent_p65_consolidated.csv

  # Specify custom output directory
  python test_visualizations.py data.csv --output-dir my_test_results

CSV File Format:
  The CSV must have these columns:
    - sequence_index: Integer index
    - sequence: Amino acid sequence string
    - motifs: Semicolon-separated list of motif patterns (e.g., "PG;PG[2]R")
        """
    )

    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to sequences CSV file (sequences_with_motifs_*.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_output',
        help='Directory to save output files (default: test_output)'
    )

    args = parser.parse_args()

    # Convert paths
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)

    # Validate CSV
    if not validate_csv(csv_path):
        logging.error("\n❌ CSV validation failed. Please check your file.")
        sys.exit(1)

    # Run tests
    success = test_visualizations(csv_path, output_dir)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
