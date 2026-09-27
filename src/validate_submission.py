"""Submission Validator Wrapper for Amazon ML Entity Resolution.

Role: Person 3 (Evaluation, Validation & Submission Integrity)

Validates output/matching_results.tsv and output/candidate_pairs.tsv
against all official competition constraints before final upload.
"""

import argparse
import os
import subprocess
import sys


def validate_submission_files(
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    check_ids: bool = False,
) -> bool:
    """Run official validation suite."""
    print("=" * 65)
    print("       AMAZON ML CHALLENGE -- SUBMISSION PRE-FLIGHT VALIDATOR       ")
    print("=" * 65)

    if not os.path.isfile(matching_file):
        print(f"[FAIL] Missing required output file: {matching_file}")
        return False

    has_candidates = os.path.isfile(candidate_file)
    if not has_candidates:
        print(f"[WARNING] Candidate file not found at: {candidate_file}")

    validator_script = os.path.join("utils", "validate_submission.py")
    if not os.path.isfile(validator_script):
        # Fallback to student_resource path
        validator_script = os.path.join("..", "6ab10eb3b23ba_student_resource", "student_resource", "utils", "validate_submission.py")

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_file,
        "--test-dir", test_dir,
    ]
    if has_candidates:
        cmd.extend(["--candidate", candidate_file])
    if check_ids:
        cmd.append("--check-ids")

    print(f"Running validation checks via: {validator_script} ...\n")
    proc = subprocess.run(cmd)

    if proc.returncode == 0:
        print("\n" + "=" * 65)
        print("  >>> [PASS] ALL SUBMISSION CHECKS PASSED SUCCESSFULLY! <<<  ")
        print("  Your files are 100% compliant with competition constraints.")
        print("=" * 65 + "\n")
        return True
    else:
        print("\n" + "=" * 65)
        print("  >>> [FAIL] SUBMISSION VALIDATION FAILED! <<<  ")
        print("  Please fix the listed errors above before submitting.")
        print("=" * 65 + "\n")
        return False




def validate_report_file(report_path: str) -> bool:
    """Validate that the evaluation report CSV contains required columns and no NaNs.
    Returns True if the file is valid, False otherwise.
    """
    import pandas as pd
    required_cols = {
        "experiment_id",
        "mode",
        "macro_f05",
        "macro_precision",
        "macro_recall",
        "total_entities",
        "singletons_accuracy",
    }
    if not os.path.isfile(report_path):
        print(f"[FAIL] Report file not found: {report_path}")
        return False
    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        print(f"[FAIL] Could not read report CSV: {e}")
        return False
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[FAIL] Report missing required columns: {missing}")
        return False
    if df.isnull().any().any():
        print("[FAIL] Report contains NaN values.")
        return False
    print("[PASS] Evaluation report validation passed.")
    return True

