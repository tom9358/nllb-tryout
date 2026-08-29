import unittest

import numpy as np
import pandas as pd

from nllb_try.config import RunConfig
from nllb_try.corpus import BaseParallelCorpus
from nllb_try.train import (
    _get_direction_mask,
    _get_model_initialization,
    get_balanced_df,
    get_training_budget,
)


def make_corpus(size: int) -> BaseParallelCorpus:
    return BaseParallelCorpus(
        "nld_Latn",
        "gos_Latn",
        pd.DataFrame(
            {
                "source_sentence": [f"source {index}" for index in range(size)],
                "target_sentence": [f"target {index}" for index in range(size)],
            }
        ),
    )


class DirectionStrategyTests(unittest.TestCase):
    def test_alternating_flips_every_row_between_epochs(self):
        frame = pd.DataFrame(
            {
                "_row_index": np.arange(8),
                "corpus_idx": [0] * 8,
            }
        )
        first_permutation = np.array([4, 2, 7, 1, 3, 0, 6, 5])
        second_permutation = np.array([1, 6, 0, 5, 2, 7, 3, 4])

        first = _get_direction_mask(
            frame, first_permutation, epoch=0, strategy="alternating"
        )
        second = _get_direction_mask(
            frame, second_permutation, epoch=1, strategy="alternating"
        )
        first_by_row = dict(zip(first_permutation, first))
        second_by_row = dict(zip(second_permutation, second))

        for row_index in frame["_row_index"]:
            self.assertNotEqual(first_by_row[row_index], second_by_row[row_index])
        self.assertEqual(int(first.sum()), 4)
        self.assertEqual(int(second.sum()), 4)

    def test_unknown_direction_strategy_is_rejected(self):
        frame = pd.DataFrame({"_row_index": [0], "corpus_idx": [0]})
        with self.assertRaisesRegex(ValueError, "direction_strategy"):
            _get_direction_mask(frame, np.array([0]), epoch=0, strategy="unsupported")


class TrainingBudgetTests(unittest.TestCase):
    def test_target_samples_sets_exact_temperature_budget(self):
        corpus = make_corpus(10)
        cfg = RunConfig(
            batch_size=4,
            num_epochs=2,
            sampling_strategy="temperature",
            target_samples_per_epoch=18,
        )

        budget = get_training_budget([corpus], cfg)
        sampled, _, _ = get_balanced_df(
            [corpus],
            sampling_strategy=cfg.sampling_strategy,
            temperature=cfg.sampling_temperature,
            target_total_samples=cfg.target_samples_per_epoch,
        )

        self.assertEqual(len(sampled), 18)
        self.assertEqual(budget["sample_counts"], [18])
        self.assertEqual(budget["steps_per_epoch"], 5)
        self.assertEqual(budget["total_steps"], 10)
        self.assertIn("_row_index", sampled.columns)

    def test_focus_cap_rejects_target_samples(self):
        cfg = RunConfig(
            sampling_strategy="focus_cap",
            focus_lang_pair=("nld_Latn", "gos_Latn"),
            target_samples_per_epoch=100,
        )
        with self.assertRaisesRegex(ValueError, "target_samples_per_epoch"):
            get_training_budget([make_corpus(10)], cfg)


class ModelInitializationTests(unittest.TestCase):
    def test_checkpoint_resume_does_not_reinitialize_new_language(self):
        cfg = RunConfig(
            modelname="facebook/base-model",
            initial_model_path="checkpoints/pooled/epoch2",
            similar_lang_nllb="nld_Latn",
        )

        model_source, similar_lang = _get_model_initialization(cfg)

        self.assertEqual(model_source, "checkpoints/pooled/epoch2")
        self.assertIsNone(similar_lang)


if __name__ == "__main__":
    unittest.main()
