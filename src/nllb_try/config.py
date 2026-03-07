from __future__ import annotations

import locale
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

locale.getpreferredencoding = lambda: "UTF-8"


@dataclass(frozen=True, slots=True)
class RunConfig:
    # Model configuration
    modelname: str = "facebook/nllb-200-distilled-600M" #'facebook/nllb-200-distilled-1.3B'
    source_langs_tatoeba: tuple[str, ...] = ("nld", "gos")
    source_langs_nllb: tuple[str, ...] = ("nld_Latn", "gos_Latn")
    new_lang_nllb: str = "gos_Latn"
    similar_lang_nllb: str = "nld_Latn"

    # Paths
    data_root_path: str = "data" # Root for all data
    tatoeba_path: str = str(Path("data") / "tatoeba")
    model_cache_path: str = "hfacemodels"

    # Run identity
    run_id: str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Training parameters
    batch_size: int = 25
    max_chars: int | None = 200
    max_length: int = 37 # Tokens
    warmup_steps: int = 110
    num_epochs: int = 12
    device: str = "cuda"

    @property
    def run_dir(self) -> str:
        model_short = self.modelname.split("/")[-1]
        langs = "-".join(self.source_langs_tatoeba)
        return str(Path("checkpoints") / f"{model_short}-{langs}-{self.run_id}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Add derived fields for convenience when persisted
        d["run_dir"] = self.run_dir
        return d


def get_default_config() -> RunConfig:
    """Factory for the default run configuration.

    No filesystem I/O is performed here.
    """
    return RunConfig()

