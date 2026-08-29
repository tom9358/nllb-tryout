import unittest
from unittest.mock import patch

import pandas as pd

from nllb_try.config import RunConfig
from nllb_try.corpus import (
    BaseParallelCorpus,
    ParallelFileCorpus,
    TatoebaCorpus,
    _global_tatoeba_split,
    _remove_tatoeba_validation_overlap,
)


class TatoebaCorpusTests(unittest.TestCase):
    @patch("nllb_try.corpus.load_tatoeba")
    @patch("nllb_try.corpus.main_download")
    def test_defers_split_until_global_split(self, main_download, load_tatoeba):
        load_tatoeba.return_value = pd.DataFrame(
            {
                "src_id": range(1, 101),
                "trg_id": range(101, 201),
                "source_sentence": [f"source {index}" for index in range(100)],
                "target_sentence": [f"target {index}" for index in range(100)],
            }
        )
        cfg = RunConfig(tatoeba_path="unused")

        corpus = TatoebaCorpus("nld", "gos", "nld_Latn", "gos_Latn", cfg)

        self.assertIsNone(corpus.df_train)
        self.assertIsNone(corpus.df_validate)
        main_download.assert_called_once_with(
            ["nld", "gos"], redownload=False, tatoeba_path="unused"
        )

        _global_tatoeba_split([corpus])

        self.assertEqual(
            list(corpus.df.columns), ["source_sentence", "target_sentence"]
        )
        self.assertEqual(len(corpus.df_train) + len(corpus.df_validate), 100)
        self.assertGreater(len(corpus.df_validate), 0)


class ValidationOverlapTests(unittest.TestCase):
    def test_removes_language_aware_overlap_from_parallel_training_only(self):
        tatoeba = TatoebaCorpus.__new__(TatoebaCorpus)
        BaseParallelCorpus.__init__(
            tatoeba,
            "nld_Latn",
            "gos_Latn",
            pd.DataFrame(
                {
                    "source_sentence": ["unused"],
                    "target_sentence": ["unused"],
                }
            ),
        )
        tatoeba.df_validate = pd.DataFrame(
            {
                "source_sentence": ["Dezelfde Nederlandse zin"],
                "target_sentence": ["Dezelfde Grunneger zin"],
            }
        )

        parallel = ParallelFileCorpus.__new__(ParallelFileCorpus)
        BaseParallelCorpus.__init__(
            parallel,
            "nld_Latn",
            "gos_Latn",
            pd.DataFrame(
                {
                    "source_sentence": ["placeholder"] * 50,
                    "target_sentence": ["placeholder"] * 50,
                }
            ),
        )
        parallel.df_train = pd.DataFrame(
            {
                "source_sentence": [
                    "  DEZELFDE   Nederlandse zin ",
                    "Zelfde tekst, verkeerde taal",
                    "Nieuwe Nederlandse zin",
                ],
                "target_sentence": [
                    "Andere Grunneger zin",
                    "Dezelfde Nederlandse zin",
                    "Nieuwe Grunneger zin",
                ],
            }
        )
        validation_before = parallel.df_validate.copy()

        removed = _remove_tatoeba_validation_overlap(parallel, [tatoeba])

        self.assertEqual(removed, 1)
        self.assertEqual(
            parallel.df_train.to_dict("records"),
            [
                {
                    "source_sentence": "Zelfde tekst, verkeerde taal",
                    "target_sentence": "Dezelfde Nederlandse zin",
                },
                {
                    "source_sentence": "Nieuwe Nederlandse zin",
                    "target_sentence": "Nieuwe Grunneger zin",
                },
            ],
        )
        pd.testing.assert_frame_equal(parallel.df_validate, validation_before)


if __name__ == "__main__":
    unittest.main()
