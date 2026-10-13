"""Regenerate existing LBO-to-RBO evaluation metrics."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
CROSS_DIR = ROOT_DIR / "rbo_prediction_using_lbo" / "cross_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CROSS_DIR))
import pipeline as original


def format_metrics(name, values):
    return [
        f"{name} MAE: {values['mae']:.6f}",
        f"{name} MSE: {values['mse']:.6f}",
        f"{name} RMSE: {values['rmse']:.6f}",
        f"{name} R2: {values['r2']:.6f}"
    ]


def format_matrix(name, matrix):
    lines = [name, "Actual\\Predicted\t" + "\t".join(original.STATE_LABELS)]
    lines.extend(
        f"{label}\t" + "\t".join(str(value) for value in row)
        for label, row in zip(original.STATE_LABELS, matrix)
    )
    return lines


def format_report(results):
    lines = [
        "LBO-to-RBO evaluation (existing model and configuration)",
        "Targets: LBO = PHI_RATIO - 1; RBO = 1 - (filename value / RBO_LIMIT)",
        "Features, XGBoost parameters, thresholds and baseline are reused from cross_dataset/pipeline.py.",
        "Per-file R2 is undefined because each RBO file has a constant target.",
        "Macro values give each file equal weight; pooled values weight rows.",
        "",
        "Pooled row-weighted metrics:",
        *format_metrics("Model", results["model"]),
        *format_metrics("LBO mean-target baseline", results["baseline"]),
        f"Baseline target: {results['baseline_target']:.6f}",
        "",
        f"Model row-level state accuracy: {results['model_state_accuracy']:.6f}",
        f"Baseline row-level state accuracy: {results['baseline_state_accuracy']:.6f}",
        "State cutoffs: Safe > 0.05; Warning > 0.025; otherwise Critical",
        *format_matrix("Model confusion matrix (rows=actual, columns=predicted):", results["model_confusion_matrix"]),
        *format_matrix("Baseline confusion matrix (rows=actual, columns=predicted):", results["baseline_confusion_matrix"]),
        "",
        "Equal-file macro MAE/MSE/RMSE:",
        f"Model: {results['macro_model_mae']:.6f}, {results['macro_model_mse']:.6f}, {results['macro_model_rmse']:.6f}",
        f"Baseline: {results['macro_baseline_mae']:.6f}, {results['macro_baseline_mse']:.6f}, {results['macro_baseline_rmse']:.6f}",
        "",
        "Per-file metrics (R2 n/a):"
    ]
    for row in results["per_file"]:
        lines.append(
            f"{row['file']} ({row['rows']} rows, target={row['target']:.6f}): "
            f"model MAE={row['model']['mae']:.6f}, MSE={row['model']['mse']:.6f}, "
            f"RMSE={row['model']['rmse']:.6f}, state accuracy={row['model_state_accuracy']:.6f}; "
            f"baseline MAE={row['baseline']['mae']:.6f}, MSE={row['baseline']['mse']:.6f}, "
            f"RMSE={row['baseline']['rmse']:.6f}, state accuracy={row['baseline_state_accuracy']:.6f}"
        )
    return "\n".join(lines) + "\n"


def evaluate():
    original.validate_config()
    lbo_runs = original.load_lbo_runs()
    rbo_runs = original.load_rbo_runs()
    original.check_sampling_compatibility(lbo_runs + rbo_runs)
    return original.evaluate(lbo_runs, rbo_runs)


def main():
    results = evaluate()
    report_path = OUTPUT_DIR / "lbo_rbo_evaluation_results.txt"
    report_path.write_text(format_report(results), encoding="utf-8")
    print(f"LBO-to-RBO evaluation report saved to {report_path}.")


class SyntheticTests(unittest.TestCase):
    def test_uses_original_target_formulas(self):
        self.assertAlmostEqual(original.lbo_target_from_ratio(1.2), 0.2)
        self.assertAlmostEqual(original.rbo_target_from_ratio(5.0, 6.3), 1 - 5 / 6.3)

    def test_report_contains_model_baseline_and_confusion_matrices(self):
        matrix = np.zeros((3, 3), dtype=int)
        metric = {"mae": 1.0, "mse": 2.0, "rmse": np.sqrt(2), "r2": -1.0}
        results = {
            "model": metric,
            "baseline": metric,
            "model_state_accuracy": 0.3,
            "baseline_state_accuracy": 0.4,
            "model_confusion_matrix": matrix,
            "baseline_confusion_matrix": matrix,
            "macro_model_mae": 1.0,
            "macro_model_mse": 2.0,
            "macro_model_rmse": np.sqrt(2),
            "macro_baseline_mae": 1.0,
            "macro_baseline_mse": 2.0,
            "macro_baseline_rmse": np.sqrt(2),
            "baseline_target": 0.1,
            "per_file": []
        }
        report = format_report(results)
        self.assertIn("Pooled row-weighted metrics", report)
        self.assertIn("Model confusion matrix", report)
        self.assertIn("LBO mean-target baseline", report)


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        main()