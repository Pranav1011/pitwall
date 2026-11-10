"""Loading fitted artifacts safely: a pickled model is only valid for the clean data
it was fitted on. fit_model.py stamps the sha256 of data/laps_clean.parquet on the
model; consumers refuse a model whose stamp no longer matches that file."""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path

from pitwall.model import MainModel


class StaleModelError(RuntimeError):
    """The fitted model was built from a different version of the clean data."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fitted_model(model_path: Path, clean_path: Path) -> MainModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    fitted_on = getattr(model, "data_hash", None)
    if fitted_on is None:
        warnings.warn(f"{model_path} has no data hash, so it cannot be checked against "
                      f"{clean_path}; re-run scripts/fit_model.py to stamp it",
                      UserWarning, stacklevel=2)
        return model
    current = file_sha256(clean_path)
    if fitted_on != current:
        raise StaleModelError(
            f"{model_path} is stale: it was fitted on {clean_path.name} with sha256 "
            f"{fitted_on[:12]}, but that file is now {current[:12]}. "
            "Re-run scripts/fit_model.py before using the model.")
    return model
