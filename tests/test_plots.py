"""Plot-generation smoke test to ensure results can be visualized."""

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

from aksha import search


def test_hits_plot_png_svg(small_fasta, small_hmm, temp_dir):
    """Run a small search and save a PNG/SVG plot of hit bitscores."""
    result = search(
        sequences=small_fasta,
        hmms=small_hmm,
        output_dir=temp_dir,
        show_progress=False,
    )

    df = result.to_dataframe()
    assert not df.empty

    # Simple bar plot of bitscores per sequence
    fig, ax = plt.subplots(figsize=(4, 3))
    df.groupby("sequence_id")["bitscore"].max().plot(kind="bar", ax=ax)
    ax.set_xlabel("sequence_id")
    ax.set_ylabel("max bitscore")
    fig.tight_layout()

    png_path = temp_dir / "hits.png"
    svg_path = temp_dir / "hits.svg"
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)

    assert png_path.exists() and png_path.stat().st_size > 0
    assert svg_path.exists() and svg_path.stat().st_size > 0
