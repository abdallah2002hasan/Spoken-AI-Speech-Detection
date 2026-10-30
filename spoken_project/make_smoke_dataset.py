"""Generate a tiny non-scientific dataset to verify the software pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
rng = np.random.default_rng(42)
t = np.arange(2 * SR) / SR
root = Path("data/smoke")
rows = []

for label, source in [(0, "real"), (1, "hifiGAN"), (1, "melgan")]:
    for speaker in range(6):
        folder = root / ("real" if label == 0 else "fake") / source / f"spk{speaker}"
        folder.mkdir(parents=True, exist_ok=True)
        for clip in range(3):
            base = np.sin(2 * np.pi * (110 + 17 * speaker) * t)
            signal = base + 0.15 * np.sin(2 * np.pi * (220 + clip * 11) * t)
            signal += 0.02 * rng.normal(size=len(t))
            if label:
                signal += 0.04 * np.sin(2 * np.pi * (4200 + 200 * clip) * t)
                signal = np.round(signal * 80) / 80

            path = folder / f"clip{clip}.wav"
            sf.write(path, signal / np.max(np.abs(signal)), SR)
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "generator": "real" if label == 0 else source,
                    # Same source utterance across real and both fake generators.
                    "group_id": f"spk{speaker}::clip{clip}",
                }
            )

pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
print("Created data/smoke + metadata.csv (software test only).")
