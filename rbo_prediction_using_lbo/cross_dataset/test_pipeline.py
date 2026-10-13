import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from unittest.mock import patch

import pipeline


class CrossDatasetPipelineTests(unittest.TestCase):
    def test_missing_expected_workbook_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(pipeline.config, "LBO_DATA_DIR", Path(folder)):
                with patch.object(pipeline, "PHI_RATIO", {"1": 1.0}):
                    with self.assertRaisesRegex(ValueError, "1 expected workbook"):
                        pipeline.load_lbo_runs()

    def test_workbook_without_target_mapping_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "2.xlsx").touch()
            with patch.object(pipeline.config, "LBO_DATA_DIR", Path(folder)):
                with patch.object(pipeline, "PHI_RATIO", {"1": 1.0}):
                    with self.assertRaisesRegex(ValueError, "missing PHI_RATIO mappings"):
                        pipeline.load_lbo_runs()

    def test_config_blocks_unconfirmed_real_run(self):
        with patch.object(pipeline.config, "LBO_SIGNAL_COLUMN", None):
            with self.assertRaisesRegex(ValueError, "set LBO_SIGNAL_COLUMN"):
                pipeline.validate_config()

    def test_rbo_filename_target_is_normalized_by_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "5.0.txt")
            np.savetxt(path, np.array([[0, 10], [1, 12], [2, 14]], dtype=float))
            runs = pipeline.load_rbo_runs(Path(folder))

        np.testing.assert_array_equal(runs[0].time, [0, 1, 2])
        np.testing.assert_array_equal(runs[0].signal, [10, 12, 14])
        self.assertAlmostEqual(runs[0].target, 5.0 / 6.3)

    def test_malformed_rbo_file_fails_without_exposing_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "5.0.txt").write_text("invalid data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "whitespace-delimited numeric"):
                pipeline.load_rbo_runs(Path(folder))

    def test_features_do_not_use_future_samples(self):
        prefix = np.arange(1, 51, dtype=float)
        first_future = np.arange(51, 61, dtype=float)
        second_future = np.arange(1001, 1011, dtype=float)
        first = pipeline.create_features(np.concatenate((prefix, first_future)))
        second = pipeline.create_features(np.concatenate((prefix, second_future)))

        pd.testing.assert_frame_equal(first.iloc[:len(prefix)], second.iloc[:len(prefix)])
        self.assertEqual(first.loc[0, "signal_lag_1"], 0)
        self.assertEqual(first.loc[0, "signal_autocorr_1"], 0)

    def test_evaluation_rejects_different_feature_schemas(self):
        first = pipeline.create_features(np.arange(60, dtype=float))
        second = first.rename(columns={"signal": "other_signal"})
        runs = [pipeline.Run(np.arange(60), np.arange(60, dtype=float), 1.0)]
        with patch.object(pipeline, "create_features", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "feature columns do not match"):
                pipeline.evaluate(runs, runs)

    def test_scaler_fits_lbo_rows_and_only_transforms_rbo_rows(self):
        class TestScaler:
            def fit_transform(self, values):
                self.fit_rows = len(values)
                return values

            def transform(self, values):
                self.transform_rows = len(values)
                return values

        class TestModel:
            def fit(self, values, targets):
                self.target_mean = np.mean(targets)
                return self

            def predict(self, values):
                return np.full(len(values), self.target_mean)

        lbo_runs = [
            pipeline.Run(np.arange(60), np.arange(60, dtype=float), 0.9),
            pipeline.Run(np.arange(60), np.arange(60, dtype=float) + 1, 1.1)
        ]
        rbo_runs = [
            pipeline.Run(np.arange(55), np.arange(55, dtype=float), 0.95),
            pipeline.Run(np.arange(55), np.arange(55, dtype=float) + 1, 1.05)
        ]
        scaler = TestScaler()
        with patch.object(pipeline, "StandardScaler", return_value=scaler):
            with patch.object(pipeline, "XGBRegressor", return_value=TestModel()):
                pipeline.evaluate(lbo_runs, rbo_runs)

        self.assertEqual(scaler.fit_rows, 120)
        self.assertEqual(scaler.transform_rows, 110)


if __name__ == "__main__":
    unittest.main()