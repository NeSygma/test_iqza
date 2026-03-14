#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["clingo"]
# ///
"""
Validation script for ASP-Bench supplementary materials.

Verifies:
- Problem set completeness (128 problems: 64 easy + 64 hard)
- Solution completeness (128 solutions: 64 easy + 64 hard)
- Solution correctness (all solutions pass ground truth validation)

Run with: uv run validate.py
"""

import json
import subprocess
import sys
from pathlib import Path


def check_problems(base_path):
    """Verify problem files."""
    print("=" * 70)
    print("CHECKING PROBLEMS")
    print("=" * 70)

    problems_dir = base_path / "problems"
    issues = []

    for difficulty in ["easy", "hard"]:
        dir_path = problems_dir / difficulty
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
            continue

        md_files = sorted(dir_path.glob(f"*_{difficulty}.md"))
        validator_files = sorted(dir_path.glob(f"*_{difficulty}_ground_truth.py"))

        print(f"\n{difficulty.upper()} problems:")
        print(f"  Descriptions: {len(md_files)}/64")
        print(f"  Validators:   {len(validator_files)}/64")

        if len(md_files) != 64:
            issues.append(f"{difficulty}: Expected 64 .md files, found {len(md_files)}")
        if len(validator_files) != 64:
            issues.append(f"{difficulty}: Expected 64 _ground_truth.py files, found {len(validator_files)}")

        # Check numbering (01-64)
        for i in range(1, 65):
            num = f"{i}" if i >= 10 else f"0{i}"
            if not any(f.name.startswith(f"{num}_") for f in md_files):
                issues.append(f"{difficulty}: Missing problem {num}")

    if not issues:
        print("\n✓ All 128 problems present (64 easy + 64 hard)")
    return issues


def check_solutions(base_path):
    """Verify solution files and validate them against ground truth."""
    print("\n" + "=" * 70)
    print("CHECKING SOLUTIONS")
    print("=" * 70)

    solutions_dir = base_path / "solutions"
    problems_dir = base_path / "problems"
    issues = []
    total_solutions = 0
    total_valid = 0

    for difficulty in ["easy", "hard"]:
        sol_dir = solutions_dir / difficulty
        prob_dir = problems_dir / difficulty

        if not sol_dir.exists():
            issues.append(f"Missing directory: {sol_dir}")
            continue

        solution_files = sorted(sol_dir.glob(f"*_{difficulty}_solution.py"))
        print(f"\n{difficulty.upper()} solutions:")
        print(f"  Files: {len(solution_files)}/64")

        if len(solution_files) != 64:
            issues.append(f"{difficulty}: Expected 64 solution files, found {len(solution_files)}")

        # Validate each solution against ground truth
        valid_count = 0
        for sol_file in solution_files:
            num = sol_file.name.split('_')[0]
            gt_files = list(prob_dir.glob(f"{num}_*_{difficulty}_ground_truth.py"))

            if not gt_files:
                issues.append(f"{difficulty}: No ground truth for {sol_file.name}")
                continue

            gt_file = gt_files[0]
            try:
                sol_result = subprocess.run(
                    [sys.executable, str(sol_file)],
                    capture_output=True, text=True, timeout=60
                )
                val_result = subprocess.run(
                    [sys.executable, str(gt_file)],
                    input=sol_result.stdout,
                    capture_output=True, text=True, timeout=30
                )
                result = json.loads(val_result.stdout)
                if result.get("valid"):
                    valid_count += 1
                else:
                    issues.append(f"{difficulty}/{sol_file.name}: validation failed")
            except Exception as e:
                issues.append(f"{difficulty}/{sol_file.name}: error - {e}")

        total_solutions += len(solution_files)
        total_valid += valid_count
        pct = valid_count * 100 // len(solution_files) if solution_files else 0
        print(f"  Valid:  {valid_count}/64 ({pct}%)")

    total_pct = total_valid * 100 // total_solutions if total_solutions else 0
    print(f"\n✓ Total: {total_valid}/{total_solutions} solutions valid ({total_pct}%)")
    return issues


def main():
    base_path = Path(__file__).parent

    print("\n" + "=" * 70)
    print("ASP-BENCH SUPPLEMENT VALIDATION")
    print("=" * 70)

    all_issues = []

    all_issues.extend(check_problems(base_path))
    all_issues.extend(check_solutions(base_path))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n❌ VALIDATION FAILED - {len(all_issues)} issues found:\n")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1
    else:
        print("\n✓ ALL CHECKS PASSED")
        print("  - 128 problems (64 easy + 64 hard)")
        print("  - 128 solutions (64 easy + 64 hard, all validated)")
        print("\nSupplement is complete and ready for distribution.")
        return 0


if __name__ == "__main__":
    exit(main())
