import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf

from nllb_try.evaluate import _evaluate, evaluate_model


class MetricDirectionTests(unittest.TestCase):
    @patch("nllb_try.evaluate.translate")
    def test_scores_translations_as_hypotheses(self, mock_translate):
        translations = ["this is complete", "another"]
        references = [
            "this is a complete reference sentence",
            "another reference sentence",
        ]
        mock_translate.return_value = translations

        bleu, chrf = _evaluate(
            source_texts=["source one", "source two"],
            references=references,
            src_lang="nld_Latn",
            tgt_lang="gos_Latn",
            model=object(),
            tokenizer=object(),
        )

        self.assertAlmostEqual(bleu, corpus_bleu(translations, [references]).score)
        self.assertAlmostEqual(chrf, corpus_chrf(translations, [references]).score)


class ValidationOnlyTests(unittest.TestCase):
    @patch("nllb_try.evaluate._calculate_metrics_for_split", return_value={})
    def test_can_skip_training_split(self, calculate_metrics):
        train = pd.DataFrame(
            {"source_sentence": ["train"], "target_sentence": ["train"]}
        )
        validate = pd.DataFrame(
            {"source_sentence": ["validate"], "target_sentence": ["validate"]}
        )
        corpus = SimpleNamespace(
            source_lang_nllb="nld_Latn",
            target_lang_nllb="gos_Latn",
            df_train=train,
            df_validate=validate,
        )

        evaluate_model(
            model=object(),
            tokenizer=object(),
            corpus_objects=[corpus],
            include_train=False,
        )

        calculate_metrics.assert_called_once()
        self.assertIs(calculate_metrics.call_args.args[0], validate)
