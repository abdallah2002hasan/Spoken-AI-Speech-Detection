from __future__ import annotations

from pathlib import Path
import re
import warnings

import pandas as pd

AUDIO_SUFFIXES = {".wav", ".flac"}
REAL_WORDS = {"real", "bonafide", "bona_fide", "human", "original", "reference"}
FAKE_WORDS = {"fake", "spoof", "generated", "synthetic", "tts", "vc"}

# Canonical generator names used internally.  Detection is substring-based so
# official WaveFake folders such as ``ljspeech_hifiGAN`` are recognized.
GENERATOR_ALIASES = {
    "parallel_wavegan": ("parallel_wavegan", "parallel-wavegan", "pwg"),
    "multi_band_melgan": ("multi_band_melgan", "multi-band-melgan"),
    "full_band_melgan": ("full_band_melgan", "full-band-melgan"),
    "melgan_large": ("melgan_large", "melgan-large"),
    "hifigan": ("hifigan", "hifi_gan", "hifi-gan"),
    "waveglow": ("waveglow",),
    "melgan": ("melgan",),
}

# WaveFake/LJSpeech generated files commonly add this suffix to the original
# utterance identifier (e.g. LJ001-0001_gen.wav -> LJ001-0001).
SOURCE_SUFFIX_RE = re.compile(r"(?:[_-](?:gen|generated|fake|synthetic))+$", re.IGNORECASE)
LJSPEECH_ID_RE = re.compile(r"^(LJ\d{3}-\d{4})", re.IGNORECASE)


def _norm_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _detect_generator(parts: list[str]) -> str | None:
    joined_parts = [_norm_token(part) for part in parts]
    # Check more-specific MelGAN names before generic melgan.
    order = [
        "parallel_wavegan",
        "multi_band_melgan",
        "full_band_melgan",
        "melgan_large",
        "hifigan",
        "waveglow",
        "melgan",
    ]
    for canonical in order:
        aliases = tuple(_norm_token(alias) for alias in GENERATOR_ALIASES[canonical])
        for part in joined_parts:
            if any(alias in part for alias in aliases):
                return canonical
    return None


def _normalize_generator(value: object) -> str:
    token = _norm_token(value)
    if token in {"", "nan", "none"}:
        return "unknown_fake"
    if token == "real":
        return "real"
    detected = _detect_generator([token])
    return detected or token


def _source_stem(path: Path) -> str:
    stem = SOURCE_SUFFIX_RE.sub("", path.stem)
    match = LJSPEECH_ID_RE.match(stem)
    if match:
        return match.group(1).upper()
    return stem


def _infer_group_id(path: Path) -> str:
    """Infer a leakage-control group from a path.

    WaveFake's LJSpeech portion is effectively single-speaker, so grouping only
    by speaker would make a meaningful split impossible.  Instead, generated
    files are paired with their underlying *source utterance* (for example
    ``LJ001-0001.wav`` and ``LJ001-0001_gen.wav`` share one group).

    For generic nested layouts, combine the immediate source folder and the
    normalized stem.  This keeps e.g. ``spk0/clip0`` paired across real/fake
    trees while avoiding collisions with ``spk1/clip0``.
    """
    source_stem = _source_stem(path)
    if LJSPEECH_ID_RE.match(source_stem):
        return source_stem
    return f"{path.parent.name}::{source_stem}"


def _validate_index(df: pd.DataFrame, root_path: Path) -> pd.DataFrame:
    required = {"path", "label", "generator", "group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset index is missing required columns: "
            f"{sorted(missing)}. Required columns are {sorted(required)}."
        )

    df = df.copy()
    df["path"] = df["path"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 (real) and 1 (AI-generated).")

    df["generator"] = df["generator"].map(_normalize_generator)
    df.loc[df["label"] == 0, "generator"] = "real"
    fake_real = (df["label"] == 1) & (df["generator"] == "real")
    if fake_real.any():
        raise ValueError("Fake rows (label=1) cannot use generator='real'.")

    df["group_id"] = df["group_id"].astype(str).str.strip()
    bad_group = df["group_id"].isin({"", "nan", "None"})
    if bad_group.any():
        raise ValueError("group_id cannot be empty. It must identify the source utterance/group.")

    if df["path"].duplicated().any():
        dup = df.loc[df["path"].duplicated(), "path"].iloc[0]
        raise ValueError(f"Duplicate audio path in dataset index: {dup}")

    missing_files = [p for p in df["path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Dataset index references missing audio files; first missing file: "
            f"{missing_files[0]}"
        )

    if df["group_id"].nunique() < 2:
        raise ValueError(
            "At least two unique group_id values are required for leakage-safe splitting. "
            "For WaveFake, group_id should be the underlying source utterance ID, not the speaker."
        )

    # Helpful diagnostic: groups appearing in both classes are desirable for
    # source-matched deepfake data and confirm that real/synthetic counterparts
    # will stay together.  Do not require this for every valid dataset.
    label_counts = df.groupby("group_id")["label"].nunique()
    paired_groups = int((label_counts > 1).sum())
    if paired_groups == 0:
        warnings.warn(
            "No group_id appears in both real and fake classes. This can be valid, but for "
            "WaveFake verify that real and generated versions of the same source utterance "
            "share exactly the same group_id.",
            RuntimeWarning,
        )

    return df.reset_index(drop=True)


def load_index(root: str) -> pd.DataFrame:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    metadata = root_path / "metadata.csv"
    if metadata.exists():
        df = pd.read_csv(metadata)

        # New, explicit format is preferred.  Accept a legacy ``speaker`` column
        # only as a migration aid; source-level group_id is safer for WaveFake.
        if "group_id" not in df.columns and "speaker" in df.columns:
            warnings.warn(
                "metadata.csv uses legacy 'speaker' grouping. Treating it as group_id. "
                "For WaveFake, replace this with a source-utterance group_id because the "
                "LJSpeech subset is effectively single-speaker.",
                RuntimeWarning,
            )
            df["group_id"] = df["speaker"]

        required = {"path", "label", "generator", "group_id"}
        if not required.issubset(df.columns):
            raise ValueError(
                "metadata.csv needs columns: path,label,generator,group_id. "
                "group_id must identify the source utterance/group shared by all related "
                "real and generated versions."
            )

        def resolve_path(value: object) -> str:
            p = Path(str(value))
            return str(p if p.is_absolute() else root_path / p)

        df["path"] = df["path"].map(resolve_path)
        return _validate_index(df, root_path)

    rows: list[dict[str, object]] = []
    for p in sorted(root_path.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_SUFFIXES:
            continue

        rel_parts = list(p.relative_to(root_path).parts[:-1])
        normalized_parts = [_norm_token(part) for part in rel_parts]
        generator = _detect_generator(rel_parts)

        has_real_marker = any(part in REAL_WORDS for part in normalized_parts)
        has_fake_marker = any(part in FAKE_WORDS for part in normalized_parts)

        if generator is not None:
            label = 1
        elif has_fake_marker and not has_real_marker:
            label = 1
            generator = "unknown_fake"
        elif has_real_marker and not has_fake_marker:
            label = 0
            generator = "real"
        else:
            # Refuse to guess ambiguous files.  Silent mislabeling is worse than
            # requiring metadata.csv for a scientific experiment.
            continue

        rows.append(
            {
                "path": str(p),
                "label": label,
                "generator": generator,
                "group_id": _infer_group_id(p),
            }
        )

    if not rows:
        raise ValueError(
            "No unambiguous labeled audio found. Organize data under explicit real/fake "
            "folders or create metadata.csv with path,label,generator,group_id."
        )

    return _validate_index(pd.DataFrame(rows), root_path)
