"""Shared, machine-independent path resolution for data preparation.

The public release never assumes a checkout layout.  Set ``NAVSIM_ROOT`` for
the raw NAVSIM files, ``DATA_ROOT`` for generated artifacts, and
``ANNOTATION_ROOT`` for reasoning-trace annotations. No home-directory
fallback is used.
"""

import os
from pathlib import Path

# Repository locations.
DATA_PROCESS_DIR = Path(__file__).resolve().parent
CODE_ROOT = DATA_PROCESS_DIR.parent
AUTODRIVE_P3_DIR = DATA_PROCESS_DIR / "AutoDrive_P3"


def _required_env(name: str, *aliases: str) -> Path:
    """Return a configured path and fail with an actionable message."""
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    names = ", ".join((name, *aliases))
    raise RuntimeError(
        f"Missing {names}. Configure the public data pipeline before running it."
    )


def navsim_root() -> Path:
    """Raw NAVSIM dataset root (contains navsim_logs/, sensor_blobs/, maps/)."""
    return _required_env("NAVSIM_ROOT")


def data_root() -> Path:
    """Output root for every pipeline artifact (processed data)."""
    return _required_env("DATA_ROOT")


def annotation_root() -> Path:
    """CoT annotation root (contains transfusion_outputs_{train,test}/)."""
    return _required_env("ANNOTATION_ROOT")


# ---------------------------------------------------------------------------
# Per-stage paths (train/test split == item)
# ---------------------------------------------------------------------------

def navsim_logs(item: str) -> Path:
    """Raw navsim log pickles for the split ("train" -> trainval, "test" -> test)."""
    log_dir = "trainval" if item == "train" else "test"
    return navsim_root() / "navsim_logs" / log_dir


def sensor_blobs(item: str) -> Path:
    """Raw camera images for the split ("train" -> trainval, "test" -> test)."""
    blob_dir = "trainval" if item == "train" else "test"
    return navsim_root() / "sensor_blobs" / blob_dir


def useful_scenes_pkl(item: str) -> Path:
    return data_root() / f"0_get_useful_{item}_scenes.pkl"


def get_json_dir(item: str) -> Path:
    return data_root() / f"1_get_json_{item}"


def frames_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"2_0_get_all_frames_{item}_{x}_{y}"


def videos_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"2_1_get_all_videos_{item}_{x}_{y}"


def traj_heading_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"3_0_get_traj_heading_{item}_{x}_{y}"


def get_v_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"3_1_get_v_{item}_{x}_{y}"


def data_4_dir(x: int, y: int) -> Path:
    return data_root() / f"4_data_{x}_{y}"


def autodrive_p3_json(item: str, x: int, y: int) -> Path:
    """AutoDrive-P3 solution JSON, optionally stored outside this checkout."""
    root = os.environ.get("AUTODRIVE_P3_ROOT")
    directory = Path(root).expanduser().resolve() if root else AUTODRIVE_P3_DIR
    return directory / f"1_RL_{item}_{x}_{y}.json"


def annotation_outputs(item: str) -> Path:
    """reasoning_trace annotation dir for the split."""
    return annotation_root() / f"transfusion_outputs_{item}"
