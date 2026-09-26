import tempfile
import unittest
from pathlib import Path

from nllb_try.csv_list_loader import find_parallel_files, load_parallel_table


class LoadParallelTableTests(unittest.TestCase):
    def test_loads_labels_and_filters_empty_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.csv"
            path.write_text(
                "\ufeffnld_Latn;gos_Latn\n"
                "Goedemorgen;Gojemörn\n"
                "   ;Lege bron\n"
                "Leeg doel;   \n",
                encoding="utf-8",
            )

            df, source_lang, target_lang = load_parallel_table(str(path))

        self.assertEqual(source_lang, "nld_Latn")
        self.assertEqual(target_lang, "gos_Latn")
        self.assertEqual(
            df.to_dict("records"),
            [{"source_sentence": "Goedemorgen", "target_sentence": "Gojemörn"}],
        )

    def test_filters_only_clear_low_quality_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.csv"
            path.write_text(
                "nld_Latn;gos_Latn\n"
                "Goedemorgen;Gojemörn\n"
                "1;1\n"
                "INHOUD:;INHOLD:\n"
                "BLADZIJDE 1;BLADZIEDE 1\n"
                "NUMMER 10 !;NUMMER 10 !\n"
                "Weerd ========;Weerd ========\n"
                "Kijk op www.voorbeeld.nl;Kiek op www.voorbeeld.nl\n"
                "Kijk op https:// www.voorbeeld.nl/pad/ vervolg/pad;Kiek op https:// www.voorbeeld.nl/pad/ vervolg/pad\n"
                "Mail info@example.nl;Mail info@example.nl\n"
                "Mail info@ example.nl;Mail info@example.nl\n"
                "\"Een &amp; twee\";\"Een &amp; twij\"\n"
                "s&lte;s&lte\n"
                "Nederlandse;Nederlandse\n"
                "Groningen;Groningen\n"
                "Kijk op https://voorbeeld.nl voor meer informatie;Kijk op https://voorbeeld.nl voor meer informatie\n"
                "\"<b>Een</b> test\";\"<b>Ain</b> test\"\n"
                "https://voorbeeld.nl;https://voorbeeld.nl\n"
                "Kijk op www.voorbeeld.nl;Kiek op de website\n"
                "HAAR;HAART\n"
                "Een zin met 13000 en 40000;N zin mit 13000 en 40000\n",
                encoding="utf-8",
            )

            df, _, _ = load_parallel_table(path)

        self.assertEqual(
            df.to_dict("records"),
            [
                {"source_sentence": "Goedemorgen", "target_sentence": "Gojemörn"},
                {
                    "source_sentence": "Kijk op [URL]",
                    "target_sentence": "Kiek op [URL]",
                },
                {
                    "source_sentence": "Kijk op [URL]",
                    "target_sentence": "Kiek op [URL]",
                },
                {
                    "source_sentence": "Mail [EMAIL]",
                    "target_sentence": "Mail [EMAIL]",
                },
                {"source_sentence": "Een & twee", "target_sentence": "Een & twij"},
                {"source_sentence": "s&lte", "target_sentence": "s&lte"},
                {"source_sentence": "Groningen", "target_sentence": "Groningen"},
                {"source_sentence": "Een test", "target_sentence": "Ain test"},
                {"source_sentence": "HAAR", "target_sentence": "HAART"},
                {
                    "source_sentence": "Een zin met 13000 en 40000",
                    "target_sentence": "N zin mit 13000 en 40000",
                },
            ],
        )

    def test_rejects_non_nllb_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.csv"
            path.write_text(
                "Nederlands;Gronings\nGoedemorgen;Gojemörn\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "NLLB language labels"):
                load_parallel_table(str(path))

    def test_infers_tab_separator_from_tsv_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.tsv"
            path.write_text(
                "nld_Latn\tgos_Latn\nGoedemorgen\tGojemörn\n",
                encoding="utf-8",
            )

            df, source_lang, target_lang = load_parallel_table(path)

        self.assertEqual((source_lang, target_lang), ("nld_Latn", "gos_Latn"))
        self.assertEqual(len(df), 1)

    def test_accepts_explicit_separator_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.csv"
            path.write_text(
                "nld_Latn,gos_Latn\nGoedemorgen,Gojemörn\n",
                encoding="utf-8",
            )

            df, _, _ = load_parallel_table(path, sep=",")

        self.assertEqual(len(df), 1)


class FindParallelFilesTests(unittest.TestCase):
    def test_expands_files_and_directories_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            csv_path = root / "one.csv"
            tsv_path = nested / "two.TSV"
            csv_path.touch()
            tsv_path.touch()
            (nested / "ignored.json").touch()

            files = find_parallel_files([root, csv_path])

        self.assertEqual(files, sorted([str(csv_path), str(tsv_path)]))

    def test_rejects_missing_paths(self):
        with self.assertRaises(FileNotFoundError):
            find_parallel_files("/path/that/does/not/exist")


if __name__ == "__main__":
    unittest.main()
