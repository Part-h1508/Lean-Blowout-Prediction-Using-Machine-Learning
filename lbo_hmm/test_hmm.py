import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluation
import model
import forecasting


class HMMTests(unittest.TestCase):
    def test_training_segments_keep_chunk_boundaries(self):
        values = np.arange(100, dtype=float)
        selected, lengths = forecasting.select_training_segments(
            values, max_samples=40, chunk_length=10
        )

        self.assertEqual(lengths, [10, 10, 10, 10])
        self.assertEqual(len(selected), sum(lengths))
        chunks = np.split(selected[:, 0], np.cumsum(lengths)[:-1])
        self.assertTrue(all(np.all(np.diff(chunk) == 1) for chunk in chunks))

    def test_forecast_uses_transition_probabilities_and_state_means(self):
        class FakeModel:
            means_ = np.array([[1.0], [-1.0]])
            transmat_ = np.array([[1.0, 0.0], [0.0, 1.0]])

        forecast = forecasting.forecast_from_state(
            FakeModel(), np.array([1.0, 0.0]), last_signal=5.0, scale=2.0, horizon=3
        )

        np.testing.assert_array_equal(forecast, [7.0, 9.0, 11.0])

    def test_causal_filter_prefix_does_not_depend_on_future_values(self):
        class FakeModel:
            n_components = 2
            startprob_ = np.array([0.5, 0.5])
            transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

            def _compute_log_likelihood(self, observations):
                values = observations[:, 0]
                return np.column_stack((np.zeros(len(values)), values))

        first = forecasting.causal_state_filter(FakeModel(), [0.0, 1.0, 0.0])
        changed_future = forecasting.causal_state_filter(FakeModel(), [0.0, 1.0, 100.0])

        np.testing.assert_allclose(first[:2], changed_future[:2])

    def test_loader_supports_single_column_tai_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "15.xlsx")
            pd.DataFrame([[1.0], [2.0], [3.0]]).to_excel(path, header=False, index=False)
            loaded = forecasting._read_run(path, "TAI")

        np.testing.assert_array_equal(loaded["signal"], [1.0, 2.0, 3.0])
        self.assertIsNone(loaded["time"])

    def test_forecast_plot_is_saved_for_available_family(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib is not installed")

        run = {
            "family": "LBO",
            "run_id": "LBO/test",
            "signal": np.arange(30, dtype=float),
        }
        predictions = pd.DataFrame({
            "family": ["LBO", "LBO"],
            "run_id": ["LBO/test", "LBO/test"],
            "origin_index": [10, 20],
            "target_index": [20, 29],
            "hmm_prediction": [19.0, 28.0],
            "persistence_prediction": [10.0, 20.0],
        })
        with tempfile.TemporaryDirectory() as folder:
            paths = forecasting.plot_family_forecasts([run], predictions, folder)

            self.assertEqual([path.name for path in paths], ["LBO_timeseries.png"])
            self.assertTrue(paths[0].is_file())

    def test_loads_headerless_time_and_signal_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "65.xlsx")
            pd.DataFrame([[0, 10], [1, 12], [2, 14]]).to_excel(path, header=False, index=False)
            sequences = model.load_sequences(folder)

        np.testing.assert_array_equal(sequences[0].time, [0, 1, 2])
        np.testing.assert_array_equal(sequences[0].signal, [10, 12, 14])

    def test_missing_time_gap_fails_sampling_check(self):
        sequence = model.Sequence("synthetic", np.array([0, 1, 2, 4]), np.arange(4, dtype=float))
        with self.assertRaisesRegex(ValueError, "missing-time gap"):
            model.check_sampling([sequence])

    def test_standardizer_uses_only_training_sequences(self):
        train = [model.Sequence("train", np.arange(3), np.array([1.0, 2.0, 3.0]))]
        test = model.Sequence("test", np.arange(3), np.array([100.0, 200.0, 300.0]))
        mean, scale = model.fit_standardizer(train)

        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(scale, np.std(train[0].signal))
        np.testing.assert_array_equal(model.standardize(test, mean, scale), [[98 / scale], [198 / scale], [298 / scale]])

    def test_metrics_distinguish_synthetic_temporal_sequences(self):
        values = np.sin(np.linspace(0, 20, 1000))
        metrics = evaluation.compare_sequences(values, values, 0.01)

        self.assertAlmostEqual(metrics["mean_absolute_difference"], 0.0)
        self.assertAlmostEqual(metrics["variance_absolute_difference"], 0.0)
        self.assertAlmostEqual(metrics["acf_mae"], 0.0)
        self.assertAlmostEqual(metrics["normalized_psd_l1"], 0.0)
        self.assertAlmostEqual(metrics["wasserstein_distance"], 0.0)

    def test_file_holdout_excludes_heldout_sequence_from_scaling(self):
        sequences = [
            model.Sequence("a", np.arange(200), np.sin(np.arange(200) / 5)),
            model.Sequence("b", np.arange(200), np.cos(np.arange(200) / 5) + 10)
        ]
        observed = []

        def fake_fit(training, mean, scale, random_state=None):
            observed.append([sequence.name for sequence in training])
            return object()

        def fake_generate(hmm, length, mean, scale, random_state=None):
            return np.full(length, mean)

        with patch.object(evaluation, "fit_hmm", side_effect=fake_fit):
            with patch.object(evaluation, "generate_sequence", side_effect=fake_generate):
                evaluation.evaluate_leave_one_file_out(sequences)

        self.assertEqual(observed, [["b"], ["a"]])

    @unittest.skipUnless(__import__("importlib").util.find_spec("hmmlearn"), "hmmlearn is not installed")
    def test_hmm_fits_and_generates_synthetic_signal(self):
        rng = np.random.default_rng(4)
        sequences = [
            model.Sequence(str(index), np.arange(500), rng.normal(index, 0.2, 500))
            for index in range(2)
        ]
        mean, scale = model.fit_standardizer(sequences)
        hmm = model.fit_hmm(sequences, mean, scale, n_states=2, random_state=4)
        generated = model.generate_sequence(hmm, 500, mean, scale, random_state=4)

        self.assertEqual(generated.shape, (500,))
        self.assertTrue(np.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()