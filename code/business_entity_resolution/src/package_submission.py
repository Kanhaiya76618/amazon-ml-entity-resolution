#!/usr/bin/env python3
"""
package_submission.py — Verify outputs with validator and build final submission zip.
"""

import os, subprocess, sys, zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT = BASE / "output"
UTILS = BASE / "utils"
DATASET = BASE / "dataset"
CODE = BASE / "code"

print("="*65)
print("RUNNING FINAL SUBMISSION VERIFICATION & PACKAGING")
print("="*65)

# 1. Run validator
print("\n[1] Running validator utils/validate_submission.py ...")
cmd = [
    sys.executable,
    str(UTILS / "validate_submission.py"),
    "--matching", str(OUTPUT / "matching_results.tsv"),
    "--candidate", str(OUTPUT / "candidate_pairs.tsv"),
    "--test-dir", str(DATASET / "test")
]
res = subprocess.run(cmd, cwd=str(BASE))
if res.returncode != 0:
    print("\nVALIDATION FAILED! Check the issues above.")
    sys.exit(1)
print("\n>>> VALIDATOR PASSED (Exit Code 0)! <<<")

# 2. Package ZIP
team_name = "team_pixel"
zip_path = BASE.parent / f"{team_name}_submission.zip"
print(f"\n[2] Assembling submission archive: {zip_path.name} ...")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    # Add output/
    for out_file in [OUTPUT / "matching_results.tsv", OUTPUT / "candidate_pairs.tsv"]:
        arcname = f"output/{out_file.name}"
        print(f"  Adding {arcname} ({out_file.stat().st_size / (1024*1024):.1f} MB)...")
        z.write(out_file, arcname=arcname)

    # Add code/business_entity_resolution/
    ber_dir = CODE / "business_entity_resolution"
    for root, dirs, files in os.walk(ber_dir):
        # Skip pycache
        if "__pycache__" in root: continue
        for f in files:
            if f.endswith((".pyc", ".pyo", ".DS_Store")): continue
            full_p = Path(root) / f
            rel_p = full_p.relative_to(CODE)
            arcname = f"code/{rel_p}"
            print(f"  Adding {arcname} ...")
            z.write(full_p, arcname=arcname)

    # Add Documentation_template.md
    doc_path = BASE / "Documentation_template.md"
    print(f"  Adding {doc_path.name} ...")
    z.write(doc_path, arcname=doc_path.name)

print("\n" + "="*65)
print(f"PACKAGE READY: {zip_path}")
print(f"Archive size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
print("="*65)
