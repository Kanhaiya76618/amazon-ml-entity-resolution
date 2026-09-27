"""
auto_finalize.py — Automated validator and zip packager to run upon inference completion.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent

def run(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(BASE))
    if res.returncode != 0:
        print(f"FAILED with exit code {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("=" * 65)
    print("AUTO-FINALIZER & SUBMISSION PACKAGER")
    print("=" * 65)
    
    # 1. Run validator
    print("[1] Validating generated TSVs against competition rules ...")
    run([
        sys.executable, "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ])
    
    # 2. Package submission ZIP
    print("\n[2] Packaging team_pixel_submission.zip ...")
    run([sys.executable, "code/business_entity_resolution/src/package_submission.py"])
    
    # 3. Update Git
    print("\n[3] Committing improvements to Git repository ...")
    subprocess.run(["git", "add", "code/", "utils/", "README.md", "Documentation_template.md"], cwd=str(BASE))
    subprocess.run(["git", "commit", "-m", "feat: multi-channel transliterated blocking v2 and LightGBM macro F0.5 optimization"], cwd=str(BASE))
    subprocess.run(["git", "push"], cwd=str(BASE))
    
    print("\n" + "=" * 65)
    print("ALL SUBMISSION ARTIFACTS VERIFIED AND PACKAGED SUCCESSFULLY!")
    print("Submission ZIP ready at: team_pixel_submission.zip")
    print("=" * 65)

if __name__ == "__main__":
    main()
