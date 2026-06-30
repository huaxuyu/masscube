from __future__ import annotations

import pickle
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def load_core_db() -> Any:
    """
    Load MassCube's internal core database.

    The result is cached, so the pickle is only read once per Python process.
    """
    db_file = resources.files("masscube").joinpath("data", "core_db.pickle")

    with db_file.open("rb") as f:
        return pickle.load(f)