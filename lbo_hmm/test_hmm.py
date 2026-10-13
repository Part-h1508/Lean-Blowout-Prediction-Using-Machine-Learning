import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluation
import model


class HMMTests(unittest.TestCase):
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