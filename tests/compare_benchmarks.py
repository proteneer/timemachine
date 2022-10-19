from glob import glob
from pathlib import Path
from typing import List, Tuple, TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Group: TypeAlias = str
Version: TypeAlias = str


def plot_benchmark_results(results: List[Tuple[Group, Version, pd.DataFrame]], metric: str, file_prefix: str):
    assert len(results) > 0

    data = pd.concat(
        {(group, version): df for group, version, df in results},
        names=["group", "version"],
    ).sort_index()

    assert data is not None

    for group, df in data.groupby("group"):
        df = df.reset_index().sort_values(["label", "version"])
        g = sns.catplot(
            data=df,
            y="label",
            x=metric,
            hue="version",
            kind="bar",
            orient="horizontal",
            legend=False,
        )

        # add bar labels
        for c in g.ax.containers:
            labels = [f"{v.get_width():.2f}" for v in c]
            g.ax.bar_label(c, labels=labels, label_type="edge")

        g.fig.set_size_inches(12, 5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(f"{file_prefix}__{group}.png")


def load_benchmark_results_json(filename: str) -> Tuple[Group, Version, pd.DataFrame]:
    df: pd.DataFrame = pd.read_json(filename).set_index(["label"])  # type: ignore
    _, group, version = filename.removesuffix(".json").split("__")
    return group, version, df


if __name__ == "__main__":
    dir = Path("benchmark_results")

    plot_benchmark_results(
        [load_benchmark_results_json(f) for f in glob(str(dir / "benchmark_results__*__*.json"))],
        metric="ns_per_day",
        file_prefix="benchmark_results",
    )

    plot_benchmark_results(
        [load_benchmark_results_json(f) for f in glob(str(dir / "benchmark_potential_results__*__*.json"))],
        metric="runs_per_second",
        file_prefix="benchmark_potential_results",
    )
