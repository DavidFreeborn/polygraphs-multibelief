"""Prepare the report sheets for direct publication on GitHub Pages."""

import argparse
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parent
ROBOTS = '<meta name="robots" content="noindex, nofollow, noarchive">'


def build(destination):
    destination = destination.resolve()
    if destination == SOURCE:
        raise ValueError("Choose an output directory separate from the source.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("results.html", "experiment-settings.html", "measures.html"):
        text = (SOURCE / name).read_text()
        marker = '<meta charset="utf-8">'
        if text.count(marker) != 1:
            raise ValueError(f"Missing or repeated charset marker in {name}")
        text = text.replace(marker, marker + ROBOTS, 1)
        (destination / name).write_text(text)
    shutil.copyfile(destination / "results.html", destination / "index.html")
    shutil.copyfile(SOURCE / "report-and-settings.zip", destination / "report-and-settings.zip")
    (destination / ".nojekyll").write_text("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
    print(f"Prepared report sheets in {args.output}")

