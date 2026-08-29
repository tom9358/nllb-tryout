from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunConfig:
    # Reproducibility
    seed: int = 9358

    # Model configuration
    modelname: str = (
        "facebook/nllb-200-distilled-600M"  #'facebook/nllb-200-distilled-1.3B'
    )
    initial_model_path: str | None = None
    source_langs_tatoeba: tuple[str, ...] = ("nld", "gos")
    source_langs_nllb: tuple[str, ...] = ("nld_Latn", "gos_Latn")
    new_lang_nllb: str = "gos_Latn"
    similar_lang_nllb: str = "nld_Latn"

    # Paths
    data_root_path: str = "data"  # Root for all data
    tatoeba_path: str = str(Path("data") / "tatoeba")
    parallel_data_paths: tuple[str, ...] = ()
    parallel_data_separator: str | None = None
    model_cache_path: str = "hfacemodels"

    # Run identity
    run_id: str = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")

    # Training parameters
    batch_size: int = 25
    learning_rate: float = 1e-4
    max_chars: int | None = 200
    max_length: int = 43  # Tokens
    warmup_steps: int = 110
    num_epochs: int = 12
    sampling_temperature: float = (
        5.0  # Temperature for balanced corpus sampling (1=proportional, inf=equal)
    )
    sampling_strategy: str = "temperature"  # temperature, focus_cap, focus_total
    focus_lang_pair: tuple[str, str] | None = None
    target_samples_per_epoch: int | None = None
    direction_strategy: str = "random"  # "random" or "alternating"
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
