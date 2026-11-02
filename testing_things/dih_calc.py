import pandas as pd
from Bio.PDB import PDBParser, calc_dihedral
import math
import argparse
import numpy as np
from natsort import natsort_key
import shutil
from pathlib import Path

def get_angle(a1, a2, a3, a4):
    if None in [a1, a2, a3, a4]:
        return None
    try:
        angle_rad = calc_dihedral(a1.get_vector(), a2.get_vector(), a3.get_vector(), a4.get_vector())
        return math.degrees(angle_rad)
    except Exception:
        return None

def get_atom(res, atom_name):
    if res and atom_name in res:
        return res[atom_name]
    return None

def calc_backbone(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('DNA', pdb_file)
    except Exception as e:
        print(f"Error reading {pdb_file}: {e}")
        return pd.DataFrame()

    # Gather all DNA residues across the whole file in order
    all_residues = []
    for model in structure:
        for chain in model:
            dna_res = [res for res in chain if res.resname.strip() in ['DA', 'DT', 'DC', 'DG', 'A', 'T', 'C', 'G']]
            all_residues.extend(dna_res)

    # Initialize the single row dictionary with the filename
    row_data = {"File": pdb_file.split("/")[-1]}

    # Loop exactly 20 times (or whatever max_residues is set to)
    for i in range(20):
        prefix = f"Res{i + 1}"  # Creates labels like Res1, Res2, etc.

        if i < len(all_residues):
            res = all_residues[i]

            # Make sure we don't calculate bonds across different chains
            prev_res = all_residues[i - 1] if (i > 0 and all_residues[i - 1].parent == res.parent) else None
            next_res = all_residues[i + 1] if (
                        i < len(all_residues) - 1 and all_residues[i + 1].parent == res.parent) else None

            p = get_atom(res, 'P')
            o5 = get_atom(res, "O5'")
            c5 = get_atom(res, "C5'")
            c4 = get_atom(res, "C4'")
            c3 = get_atom(res, "C3'")
            o3 = get_atom(res, "O3'")

            prev_o3 = get_atom(prev_res, "O3'")
            next_p = get_atom(next_res, 'P')
            next_o5 = get_atom(next_res, "O5'")

            # Store the 6 angles for this specific residue number
            row_data[f"{prefix}_Alpha"] = get_angle(prev_o3, p, o5, c5)
            row_data[f"{prefix}_Beta"] = get_angle(p, o5, c5, c4)
            row_data[f"{prefix}_Gamma"] = get_angle(o5, c5, c4, c3)
            row_data[f"{prefix}_Delta"] = get_angle(c5, c4, c3, o3)
            row_data[f"{prefix}_Epsilon"] = get_angle(c4, c3, o3, next_p)
            row_data[f"{prefix}_Zeta"] = get_angle(c3, o3, next_p, next_o5)
        else:
            # If the file has fewer than 20 residues, pad the rest of the columns with NaN
            row_data[f"{prefix}_Alpha"] = np.nan
            row_data[f"{prefix}_Beta"] = np.nan
            row_data[f"{prefix}_Gamma"] = np.nan
            row_data[f"{prefix}_Delta"] = np.nan
            row_data[f"{prefix}_Epsilon"] = np.nan
            row_data[f"{prefix}_Zeta"] = np.nan

    # Return this single row wrapped in a DataFrame
    return pd.DataFrame([row_data])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DNA backbone dihedral angles from a PDB file.")
    parser.add_argument('-i',"--pdb_file", help="Path to the input PDB file", nargs='+')
    parser.add_argument('-o',"--output", help="Path to the output .csv file")
    args = parser.parse_args()

    all_df=[]
    for pdb in args.pdb_file:
        df=calc_backbone(pdb)
        if not df.empty:
            all_df.append(df)
    if all_df:
        master_df = pd.concat(all_df, ignore_index=True)
        master_df = master_df.drop(columns=['Res1_Alpha', 'Res20_Zeta'])
        master_df = master_df.sort_values(by=['File'], key=natsort_key)
        print(master_df.to_string())
        if args.output:
            master_df.to_csv(args.output, index=False)
            print(f"\nData successfully saved to {args.output}")




