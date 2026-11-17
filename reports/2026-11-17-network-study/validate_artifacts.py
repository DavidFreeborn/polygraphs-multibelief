"""Validate offline HTML images, local links, figures and report tables."""

import base64
import io
import json
from html.parser import HTMLParser
from pathlib import Path
from PIL import Image
import pandas as pd

HERE = Path(__file__).resolve().parent


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = 0
        self.links = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "img":
            source = attributes["src"]
            assert source.startswith("data:image/png;base64,")
            with Image.open(
                io.BytesIO(base64.b64decode(source.split(",", 1)[1]))
            ) as image:
                image.verify()
            assert attributes.get("alt")
            self.images += 1
        if tag == "a":
            self.links.append(attributes["href"])


def main():
    parser = ReportParser()
    parser.feed((HERE / "results.html").read_text(encoding="utf-8"))
    assert parser.images == 81, parser.images
    for link in parser.links:
        if not link.startswith(("http", "#", "mailto:")):
            assert (HERE / link).exists(), link
    assert len(list((HERE / "figures").glob("*.png"))) == 81
    assert len(list((HERE / "figures").glob("*.svg"))) == 81
    m = pd.read_csv(HERE / "data/metrics.csv")
    n = pd.read_csv(HERE / "data/nodes.csv")
    assert m.groupby("id").seed.nunique().eq(20).all()
    assert m.groupby(["id", "seed"])["round"].max().eq(500).all()
    assert m.groupby(["id", "seed"])["round"].min().eq(0).all()
    assert not m.duplicated(["id", "seed", "round"]).any()
    result = dict(
        embedded_figures=parser.images,
        metric_rows=len(m),
        node_rows=len(n),
        conditions=m.id.nunique(),
        local_links="verified",
        status="passed",
    )
    (HERE / "data/artifact-validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
