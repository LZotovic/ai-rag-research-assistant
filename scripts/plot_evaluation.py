"""Turn the RAG evaluation CSV into a presentation-ready dashboard."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_arguments():
    """Let the same script work with default or custom result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/rag_evaluation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation_dashboard.png"),
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    """CSV stores booleans as text, so convert them before averaging."""
    return value.strip().lower() in {"1", "true", "yes"}


def main():
    args = parse_arguments()
    with args.input.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"No evaluation rows found in {args.input}")

    # Recall@k is the fraction of questions whose expected page appeared in
    # the first k retrieved chunks.
    recall = [
        sum(int(row[f"hit_at_{k}"]) for row in rows) / len(rows) * 100
        for k in (1, 3, 5)
    ]
    retrieval_ms = [float(row["retrieval_ms"]) for row in rows]
    generation_ms = [
        float(row["generation_ms"])
        for row in rows
        if row["generation_ms"].strip()
    ]
    scores = [float(row["top_score"]) for row in rows]

    # A 2x2 dashboard makes the most important evidence readable in one image.
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle("CiteWise RAG Evaluation", fontsize=18, fontweight="bold")

    axes[0, 0].bar(["Recall@1", "Recall@3", "Recall@5"], recall,
                   color=["#5b8def", "#55b89a", "#f0aa4f"])
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].set_ylabel("Questions retrieved (%)")
    axes[0, 0].set_title("Retrieval accuracy")
    for position, value in enumerate(recall):
        axes[0, 0].text(position, value + 2, f"{value:.0f}%", ha="center")

    labels = [f"Q{number}" for number in range(1, len(rows) + 1)]
    axes[0, 1].bar(labels, retrieval_ms, color="#5b8def")
    axes[0, 1].set_ylabel("Milliseconds")
    axes[0, 1].set_title("Semantic retrieval latency")

    axes[1, 0].bar(labels, scores, color="#8d72cc")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Cosine similarity")
    axes[1, 0].set_title("Top semantic-search score")

    # Generation values are absent only when evaluation used --skip-generation.
    if generation_ms:
        axes[1, 1].bar(labels[: len(generation_ms)], generation_ms,
                       color="#f0aa4f")
        axes[1, 1].set_ylabel("Milliseconds")
        axes[1, 1].set_title("Local LLM generation latency")
    else:
        axes[1, 1].text(0.5, 0.5, "Generation was skipped", ha="center",
                        va="center", fontsize=13)
        axes[1, 1].set_title("Local LLM generation latency")

    # Add two answer-quality summaries when generation results are available.
    grounded = [row["grounded"] for row in rows if row["grounded"].strip()]
    citations = [
        row["citation_valid"] for row in rows if row["citation_valid"].strip()
    ]
    if grounded and citations:
        grounded_rate = sum(as_bool(value) for value in grounded) / len(grounded)
        citation_rate = sum(as_bool(value) for value in citations) / len(citations)
        figure.text(
            0.5,
            0.01,
            f"LLM marked grounded: {grounded_rate:.0%}   |   "
            f"Valid citations (grounded): {citation_rate:.0%}   |   "
            f"Questions: {len(rows)}",
            ha="center",
            fontsize=11,
        )

    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Wrote evaluation dashboard to {args.output}")


if __name__ == "__main__":
    main()
