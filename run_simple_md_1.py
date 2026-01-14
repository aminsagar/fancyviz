import warnings
import io
import gemmi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdbfixer import PDBFixer
import argparse
from datetime import datetime
from pathlib import Path
from MDAnalysis.analysis import rms, align
import time
import pandas as pd
import subprocess
import MDAnalysis as mda
import numpy as np
import sys
import shutil
from copy import deepcopy
from openmmforcefields.generators import EspalomaTemplateGenerator
from simtk import unit
from openmm.app import Modeller
from sys import stdout
from openmm.unit import *
from openmm import *
from openmm.app import *
from openff.toolkit.topology import Molecule
import espaloma as esp
import torch
from pathlib import Path as Path2
from Bio import PDB
from rdkit import Chem
import os
from Bio.PDB import Superimposer

warnings.filterwarnings("ignore")

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_step(step_num, total_steps, description):
    """Print a formatted step indicator"""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 80)

def create_simplified_file_structure(base_dir):
    """Creates a simplified directory structure for the simulation."""
    directories = {
        "structures": os.path.join(base_dir, "structures"),
        "md": os.path.join(base_dir, "md"),
        "analysis": os.path.join(base_dir, "analysis")
    }
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
    return directories

def convert_cif_to_pdb(cif_path, pdb_path):
    """Converts a CIF file to a PDB file using gemmi."""
    print_step(1, 8, f"Converting {cif_path} to PDB format")
    doc = gemmi.cif.read_file(cif_path)
    st = gemmi.make_structure_from_block(doc.sole_block())
    st.write_pdb(pdb_path)
    print(f"✓ Converted to {pdb_path}")
    return pdb_path

def split_pdb(input_pdb, out_receptor, out_ligand, receptor_chains, ligand_chain):
    """Splits a PDB file into receptor and ligand files based on chain IDs."""
    print_step(2, 8, f"Splitting {input_pdb} into receptor and ligand chains")
    u = mda.Universe(input_pdb)
    
    receptor_sel_str = " or ".join([f"chainid {c.strip()}" for c in receptor_chains.split(',')])
    receptor = u.select_atoms(receptor_sel_str)
    ligand = u.select_atoms(f"chainid {ligand_chain.strip()}")

    if receptor.n_atoms == 0:
        raise ValueError(f"Receptor chain(s) '{receptor_chains}' not found or empty in {input_pdb}.")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand chain '{ligand_chain}' not found or empty in {input_pdb}.")

    receptor.write(out_receptor)
    ligand.write(out_ligand)
    print(f"✓ Receptor ({receptor.n_residues} residues) saved to {out_receptor}")
    print(f"✓ Ligand ({ligand.n_residues} residues) saved to {out_ligand}")

def fix_receptor_with_pdbfixer(receptor_pdb, fixed_receptor_pdb):
    """Fixes a protein PDB file using PDBFixer."""
    print_step(3, 8, f"Fixing receptor {receptor_pdb} with PDBFixer")
    fixer = PDBFixer(filename=receptor_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False) # Keep water for now, will be removed by parameterizer
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_receptor_pdb, 'w'))
    print(f"✓ Fixed receptor saved to {fixed_receptor_pdb}")

def prepare_ligand_with_yasara(ligand_pdb, output_mol, yasara_path):
    """Adds hydrogens to a ligand PDB and saves it as a MOL file using YASARA."""
    print_step(4, 8, f"Preparing ligand {ligand_pdb} with YASARA")
    mcr_path = Path(ligand_pdb).with_suffix(".mcr")
    
    yasara_abs_path = Path(yasara_path).resolve()
    ligand_pdb_abs = Path(ligand_pdb).resolve()
    output_mol_abs = Path(output_mol).resolve()

    mcr_content = f"""
OnError Exit
LoadPDB "{ligand_pdb_abs}"
AddHydAll
SaveMol All, "{output_mol_abs}"
Exit
"""
    with open(mcr_path, 'w') as f:
        f.write(mcr_content)

    cmd = f'"{yasara_abs_path}" -txt "{mcr_path}"'
    print(f"Running YASARA command...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=300)
        print("✓ YASARA executed successfully.")
        if result.stdout:
            print("YASARA output:\n" + result.stdout)
    except subprocess.CalledProcessError as e:
        print("YASARA execution failed.")
        print("Stderr:", e.stderr)
        print("Stdout:", e.stdout)
        raise
    except subprocess.TimeoutExpired as e:
        print("YASARA timed out.")
        print("Stderr:", e.stderr)
        print("Stdout:", e.stdout)
        raise

def parameterize_complex(ligand_mol_path, protein_pdb, dirs):
    """Parameterizes the protein-ligand complex."""
    print_step(5, 8, "Parameterizing complex with Espaloma and AMBER")
    
    print("🔹 Loading ligand structure...")
    molecule = Molecule.from_file(ligand_mol_path, allow_undefined_stereo=True)
    
    print("🔹 Setting up force fields...")
    amber_forcefields = ['amber/protein.ff14SB.xml', 'amber14/tip3pfb.xml']
    forcefield = ForceField(*amber_forcefields)
    espaloma_generator = EspalomaTemplateGenerator(molecules=molecule, forcefield='espaloma-0.3.1')
    
    print("🔹 Applying Espaloma parameterization...")
    openmm_topology = molecule.to_topology().to_openmm()
    openmm_positions = molecule.conformers[0].to_openmm()
    forcefield.registerTemplateGenerator(espaloma_generator.generator)
    
    print("🔹 Loading and parameterizing protein...")
    protein = PDBFile(protein_pdb)
    protein_modeller = Modeller(protein.topology, protein.positions)
    protein_modeller.add(openmm_topology, openmm_positions)
    
    print("🔹 Adding solvent and ions...")
    protein_modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer, 
                              boxShape='dodecahedron', ionicStrength=0.15*molar)
    print(f"✓ System setup complete with {protein_modeller.topology.getNumAtoms():,} atoms")
    
    complex_pdb_path = os.path.join(dirs["structures"], "complex_solvated.pdb")
    with open(complex_pdb_path, 'w') as outfile:
        PDBFile.writeFile(protein_modeller.topology, protein_modeller.positions, outfile)
    print(f"✓ Solvated complex saved to {complex_pdb_path}")
    
    system = forcefield.createSystem(protein_modeller.topology, 
                                   nonbondedMethod=PME,
                                   nonbondedCutoff=0.9*nanometer, 
                                   constraints=HBonds,
                                   hydrogenMass=1.5*amu)
    
    return protein_modeller.topology, protein_modeller.positions, system, complex_pdb_path

def get_protein_backbone_indices(complex_pdb_path, receptor_chains):
    """Gets backbone atom indices for the protein."""
    u = mda.Universe(complex_pdb_path)
    receptor_sel_str = " or ".join([f"chainid {c.strip()}" for c in receptor_chains.split(',')])
    protein_bb_sel = u.select_atoms(f"({receptor_sel_str}) and backbone")
    return protein_bb_sel.indices

def add_posres(system, positions, restraint_force, restraint_indices):
    """Adds position restraints to the system."""
    force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force_amount = restraint_force * kilocalories_per_mole/angstroms**2
    force.addGlobalParameter("k", force_amount)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    print(f"🔹 Adding restraints on {len(restraint_indices)} atoms.")
    for i in restraint_indices:
        force.addParticle(i, positions[i].value_in_unit(nanometers))
    posres_sys = deepcopy(system)
    posres_sys.addForce(force)
    return posres_sys

def run_simulation_protocol(topology, positions, system, dirs, restraint_indices, nvt_steps=500, npt_steps=500, prod_steps=1000):
    """Runs the full MD simulation protocol."""
    print_step(6, 8, "Running MD Simulation")
    
    temperature = 300 * unit.kelvin
    collision_rate = 1.0 / unit.picoseconds
    timestep = 2.0 * unit.femtoseconds
    
    integrator = LangevinMiddleIntegrator(temperature, collision_rate, timestep)
    platform = Platform.getPlatformByName('CUDA')
    
    posres_sys = add_posres(system, positions, 100, restraint_indices)
    
    simulation = Simulation(topology, posres_sys, integrator, platform)
    simulation.context.setPositions(positions)
    
    print("🔹 Minimizing energy...")
    simulation.minimizeEnergy()
    min_positions = simulation.context.getState(getPositions=True).getPositions()
    min_pdb_path = os.path.join(dirs["md"], "minimized.pdb")
    PDBFile.writeFile(simulation.topology, min_positions, open(min_pdb_path, 'w'))
    
    dcd_path = os.path.join(dirs["md"], "trajectory.dcd")
    log_path = os.path.join(dirs["md"], "simulation.log")
    
    simulation.reporters.append(DCDReporter(dcd_path, 1000))
    simulation.reporters.append(StateDataReporter(log_path, 1000,
        step=True, potentialEnergy=True, temperature=True, progress=True,
        remainingTime=True, speed=True, totalSteps=nvt_steps+npt_steps+prod_steps))
            
    print("🔹 Running NVT equilibration...")
    simulation.step(nvt_steps)
    
    print("🔹 Adding barostat and running NPT equilibration...")
    posres_sys.addForce(MonteCarloBarostat(1*unit.bar, temperature))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(npt_steps)
    
    print("🔹 Running production (restraints released)...")
    simulation.context.setParameter('k', 0)
    integrator.setStepSize(4.0 * femtoseconds)
    simulation.step(prod_steps)

    print("✓ Simulation finished.")
    return min_pdb_path, dcd_path

def process_trajectory(topology_pdb, trajectory_dcd, dirs, ligand_chain):
    """Strips water and ions from the trajectory and creates a matching topology."""
    print_step(7, 8, "Processing Trajectory")
    processed_dcd_path = os.path.join(dirs["md"], "trajectory_processed.dcd")
    cpptraj_script_path = os.path.join(dirs["md"], "process.in")

    u_initial_topology = mda.Universe(topology_pdb)
    ligand_atoms_initial = u_initial_topology.select_atoms(f"chainid {ligand_chain}")
    if ligand_atoms_initial.n_residues == 0:
        raise ValueError(f"Could not identify ligand in chain {ligand_chain} from topology {topology_pdb}")
    ligand_resname = ligand_atoms_initial.residues.resnames[0]


    script = f"""
parm {topology_pdb}
trajin {trajectory_dcd}
autoimage
strip :HOH,NA,CL
trajout {processed_dcd_path}
go
"""
    with open(cpptraj_script_path, 'w') as f:
        f.write(script)
    
    print("🔹 Running cpptraj to strip solvent...")
    result = subprocess.run(f"cpptraj -i {cpptraj_script_path}", shell=True, check=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("cpptraj stderr:", result.stderr)
        print("cpptraj stdout:", result.stdout)
        raise subprocess.CalledProcessError(result.returncode, f"cpptraj -i {cpptraj_script_path}", output=result.stdout, stderr=result.stderr)

    print(f"✓ Processed trajectory saved to {processed_dcd_path}")

    processed_topology_pdb = os.path.join(dirs["md"], "processed_topology.pdb")
    
    u_for_processed_topology = mda.Universe(topology_pdb)
    protein_and_ligand = u_for_processed_topology.select_atoms(f"protein or resname {ligand_resname}")
    
    protein_and_ligand.write(processed_topology_pdb)
    print(f"✓ Processed topology saved to {processed_topology_pdb}")

    return processed_topology_pdb, processed_dcd_path

def analyze_trajectory_stability(topology_pdb, trajectory_dcd, dirs, receptor_chains, ligand_chain):
    """Performs RMSD and RMSF analysis."""
    print_step(8, 8, "Analyzing Trajectory Stability")
    u = mda.Universe(topology_pdb, trajectory_dcd)
    
    analysis_dir = dirs["analysis"]

    receptor_sel_str = " or ".join([f"chainid {c.strip()}" for c in receptor_chains.split(',')])
    
    aligner = align.AlignTraj(u, u, select=f"({receptor_sel_str}) and backbone", in_memory=True).run()
    
    receptor_bb = u.select_atoms(f"({receptor_sel_str}) and backbone")
    ligand_heavy = u.select_atoms(f"chainid {ligand_chain} and not name H*")
    
    R_receptor = rms.RMSD(receptor_bb, receptor_bb, ref_frame=0).run()
    R_ligand = rms.RMSD(ligand_heavy, ligand_heavy, ref_frame=0).run()

    plt.figure()
    plt.plot(R_receptor.results.rmsd[:,0], R_receptor.results.rmsd[:,2], label="Receptor Backbone")
    plt.plot(R_ligand.results.rmsd[:,0], R_ligand.results.rmsd[:,2], label="Ligand Heavy Atoms")
    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD (Å)")
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, "rmsd.png"))
    plt.close()

    receptor_ca = u.select_atoms(f"({receptor_sel_str}) and name CA")
    F_receptor = rms.RMSF(receptor_ca).run()

    plt.figure()
    plt.plot(F_receptor.results.rmsf)
    plt.xlabel("Residue Index")
    plt.ylabel("RMSF (Å)")
    plt.savefig(os.path.join(analysis_dir, "rmsf_receptor.png"))
    plt.close()
    
    print(f"✓ Analysis plots saved in {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run a simple MD simulation pipeline for a protein-ligand complex.")
    parser.add_argument("--input_file", required=True, help="Path to the input PDB or CIF file containing the complex.")
    parser.add_argument("--receptor_chains", required=True, help="Comma-separated chain ID(s) for the receptor (e.g., 'A' or 'A,B').")
    parser.add_argument("--ligand_chain", required=True, help="Chain ID for the ligand.")
    parser.add_argument("--yasara_path", required=True, help="Path to the YASARA executable.")
    parser.add_argument("--output_dir", default="md_output", help="Directory to save all simulation outputs.")
    args = parser.parse_args()

    print_section_header("MD Simulation Pipeline Started")
    
    original_cwd = os.getcwd()
    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    os.chdir(output_dir_abs)
    
    try:
        dirs = create_simplified_file_structure(".")

        input_file_path = os.path.abspath(os.path.join(original_cwd, args.input_file))
        file_extension = Path(input_file_path).suffix.lower()
    
        if file_extension == ".cif":
            pdb_path = os.path.join(dirs["structures"], "converted_from_cif.pdb")
            input_file_path = convert_cif_to_pdb(input_file_path, pdb_path)

        receptor_pdb = os.path.join(dirs["structures"], "receptor.pdb")
        ligand_pdb = os.path.join(dirs["structures"], "ligand.pdb")
        split_pdb(input_file_path, receptor_pdb, ligand_pdb, args.receptor_chains, args.ligand_chain)
        
        fixed_receptor_pdb = os.path.join(dirs["structures"], "receptor_fixed.pdb")
        fix_receptor_with_pdbfixer(receptor_pdb, fixed_receptor_pdb)
        
        ligand_mol = os.path.join(dirs["structures"], "ligand.mol")
        prepare_ligand_with_yasara(ligand_pdb, ligand_mol, args.yasara_path)

        topology, positions, system, complex_pdb = parameterize_complex(ligand_mol, fixed_receptor_pdb, dirs)
        
        backbone_indices = get_protein_backbone_indices(complex_pdb, args.receptor_chains)

        min_pdb, dcd_file = run_simulation_protocol(topology, positions, system, dirs, backbone_indices)
        
        processed_topology_pdb, processed_dcd = process_trajectory(min_pdb, dcd_file, dirs, args.ligand_chain)
        
        analyze_trajectory_stability(processed_topology_pdb, processed_dcd, dirs, args.receptor_chains, args.ligand_chain)

        print_section_header("Pipeline Finished Successfully")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
