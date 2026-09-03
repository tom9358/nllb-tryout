import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
from run_kreuze_multilingual_continuation import get_final_checkpoint
from run_kreuze_multilingual_experiment import _get_pooled_focus_size


def make_corpus(size: int) -> BaseParallelCorpus:
    return make_named_corpus("nld_Latn", "gos_Latn", size)


def make_named_corpus(
    source_lang: str, target_lang: str, size: int
) -> BaseParallelCorpus:
    return BaseParallelCorpus(
        source_lang,
        target_lang,
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

    def test_focus_total_splits_exact_budget_between_focus_and_rest(self):
        corpora = [
            make_named_corpus("nld_Latn", "gos_Latn", 10),
            make_named_corpus("nld_Latn", "eng_Latn", 100),
            make_named_corpus("gos_Latn", "eng_Latn", 25),
            make_named_corpus("eng_Latn", "deu_Latn", 4),
        ]
        cfg = RunConfig(
            batch_size=8,
            sampling_strategy="focus_total",
            focus_lang_pair=("nld_Latn", "gos_Latn"),
            sampling_temperature=5.0,
            target_samples_per_epoch=40,
        )

        budget = get_training_budget(corpora, cfg)
        sampled, _, _ = get_balanced_df(
            corpora,
            sampling_strategy=cfg.sampling_strategy,
            focus_lang_pair=cfg.focus_lang_pair,
            temperature=cfg.sampling_temperature,
            target_total_samples=cfg.target_samples_per_epoch,
        )

        self.assertEqual(budget["sample_counts"][0], 20)
        self.assertEqual(sum(budget["sample_counts"][1:]), 20)
        self.assertEqual(budget["samples_per_epoch"], 40)
        self.assertEqual(len(sampled), 40)

    def test_multilingual_target_uses_complete_pooled_focus_size(self):
        corpora = [
            make_named_corpus("nld_Latn", "gos_Latn", 10),
            make_named_corpus("nld_Latn", "gos_Latn", 90),
            make_named_corpus("nld_Latn", "eng_Latn", 1_000),
        ]

        expected = len(corpora[0].df_train) + len(corpora[1].df_train)
        self.assertEqual(_get_pooled_focus_size(corpora), expected)


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

    def test_multilingual_continuation_uses_configured_final_epoch(self):
        with TemporaryDirectory() as temp_dir:
            source_run = Path(temp_dir)
            expected = source_run / "checkpoints" / "epoch6"
            expected.mkdir(parents=True)

            checkpoint = get_final_checkpoint(source_run, {"num_epochs": 6})

            self.assertEqual(checkpoint, expected)


if __name__ == "__main__":
    unittest.main()
