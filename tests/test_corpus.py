import unittest
from unittest.mock import patch

import pandas as pd

from nllb_try.config import RunConfig
from nllb_try.corpus import TatoebaCorpus, _global_tatoeba_split


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


if __name__ == "__main__":
    unittest.main()
