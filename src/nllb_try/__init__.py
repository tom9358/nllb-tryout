"""nllb_try package.

Public API:
- RunConfig: structured run configuration
- get_default_config: factory for default RunConfig
"""

from .config import RunConfig, get_default_config

__all__ = ["RunConfig", "get_default_config"]
