import argparse
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
import os, shutil
from pathlib import Path



def convert_cif_to_pdb(cif_file, pdb_file):
    # 1. Read the CIF file
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", cif_file)
    except Exception as e:
        print(f"Error reading CIF file: {e}")
        return

    # 2. Write the PDB file
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(pdb_file)
        print(f"Successfully converted {cif_file} to {pdb_file}")
    except Exception as e:
        print(f"Error saving PDB file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an mmCIF file to a PDB file.")
    parser.add_argument('-i',"--input_cif", help="Path to the input .cif file", required=True)
    parser.add_argument('-o',"--output", help="Path to the output folder")
    args = parser.parse_args()
    out_path = Path(args.output)
    if out_path.exists() and out_path.is_dir():
        shutil.rmtree(out_path)
        print(f"Output folder exists, deleting {out_path}")

    input_fold=Path(args.input_cif)
    os.makedirs(args.output, exist_ok=True)

    i = 1
    for inp in input_fold.rglob("*.cif"):
        out = os.path.join(args.output, f"file_{i}.pdb")
        convert_cif_to_pdb(inp, out)
        i += 1