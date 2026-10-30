from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    import pandas as pd
except Exception as e:  # pragma: no cover - helpful error if pandas is missing
    raise ImportError(
        "pandas is required to run this script. Install it with 'pip install pandas'."
    ) from e

AUDIO_SUFFIXES = {".wav", ".flac"}
ID_RE = re.compile(r"(LJ\d{3}-\d{4})", re.IGNORECASE)

GENERATORS = {
    "hifigan": "ljspeech_hifiGAN",
    "melgan": "ljspeech_melgan",
    "multi_band_melgan": "ljspeech_multi_band_melgan",
    "parallel_wavegan": "ljspeech_parallel_wavegan",
    "waveglow": "ljspeech_waveglow",
}


def source_id(path: Path) -> str | None:
    match = ID_RE.search(path.stem)
    return match.group(1).upper() if match else None


def index_audio(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        sid = source_id(p)
        if sid is None:
            continue
        if sid in result:
            raise ValueError(f"Duplicate source ID {sid} in {folder}:\n{result[sid]}\n{p}")
        result[sid] = p.resolve()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create leakage-safe metadata.csv for LJSpeech + WaveFake."
    )
    ap.add_argument("--real-root", required=True, help="Path to LJSpeech-1.1/wavs")
    ap.add_argument(
        "--generated-root", required=True, help="Path to WaveFake generated_audio"
    )
    ap.add_argument(
        "--output", default="data/wavefake/metadata.csv", help="Output CSV path"
    )
    ap.add_argument(
        "--max-source-ids",
        type=int,
        default=None,
        help="Optional source-utterance limit for a smaller experiment. Default: all matched IDs.",
    )
    args = ap.parse_args()

    real_root = Path(args.real_root).expanduser().resolve()
    generated_root = Path(args.generated_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    if not real_root.is_dir():
        raise FileNotFoundError(f"Real-audio folder not found: {real_root}")
    if not generated_root.is_dir():
        raise FileNotFoundError(f"WaveFake folder not found: {generated_root}")

    real = index_audio(real_root)
    if not real:
        raise ValueError(f"No LJSpeech audio IDs found in {real_root}")

    fake_indexes: dict[str, dict[str, Path]] = {}
    for generator, folder_name in GENERATORS.items():
        folder = generated_root / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected WaveFake folder is missing: {folder}\n"
                f"Do not rename the official generator folders."
            )
        fake_indexes[generator] = index_audio(folder)
        if not fake_indexes[generator]:
            raise ValueError(f"No LJSpeech IDs found in {folder}")

    common_ids = set(real)
    for index in fake_indexes.values():
        common_ids &= set(index)
    common_ids = sorted(common_ids)

    if not common_ids:
        raise ValueError(
            "No common LJSpeech utterance IDs were found across real audio and all selected generators."
        )

    if args.max_source_ids is not None:
        if args.max_source_ids < 2:
            raise ValueError("--max-source-ids must be at least 2")
        common_ids = common_ids[: args.max_source_ids]

    rows: list[dict[str, object]] = []
    for sid in common_ids:
        rows.append(
            {
                "path": str(real[sid]),
                "label": 0,
                "generator": "real",
                "group_id": sid,
            }
        )
        for generator, index in fake_indexes.items():
            rows.append(
                {
                    "path": str(index[sid]),
                    "label": 1,
                    "generator": generator,
                    "group_id": sid,
                }
            )

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Created: {out}")
    print(f"Source utterances/groups: {len(common_ids)}")
    print(f"Total audio rows: {len(df)}")
    print("\nCounts:")
    print(df.groupby(["label", "generator"]).size().to_string())
    print("\nLeakage grouping check:")
    per_group = df.groupby("group_id").agg(rows=("path", "size"), classes=("label", "nunique"))
    print(f"Groups containing both real and fake: {(per_group['classes'] == 2).sum()} / {len(per_group)}")
    print(f"Expected rows per complete group: {1 + len(GENERATORS)}")


if __name__ == "__main__":
    main()
