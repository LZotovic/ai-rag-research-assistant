"""Run retrieval and local-generation evaluation and write plot-ready CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

from rag_assistant.answers import OllamaAnswerGenerator, OllamaUnavailableError
from rag_assistant.embeddings import SentenceTransformerEmbedder
from rag_assistant.index import DocumentIndex
from rag_assistant.pdf import extract_pdf_pages


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, help="PDF to evaluate")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/ecg_questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rag_evaluation.csv"),
    )
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    cases = json.loads(args.questions.read_text(encoding="utf-8"))

    pages = extract_pdf_pages(args.pdf.read_bytes())
    index = DocumentIndex(SentenceTransformerEmbedder())
    index.add_document(args.pdf.name, pages)
    generator = OllamaAnswerGenerator(
        model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    rows = []
    for case in cases:
        retrieval_start = perf_counter()
        results = index.search(case["question"], top_k=5)
        retrieval_ms = (perf_counter() - retrieval_start) * 1000
        retrieved_pages = [result.chunk.page for result in results]

        generation_ms = None
        grounded = None
        citation_valid = None
        answer_text = ""
        if not args.skip_generation:
            generation_start = perf_counter()
            try:
                answer = generator.generate(case["question"], results)
                generation_ms = (perf_counter() - generation_start) * 1000
                grounded = answer.grounded
                cited_pages = [citation.page for citation in answer.citations]
                citation_valid = all(page in retrieved_pages for page in cited_pages)
                answer_text = answer.text
            except OllamaUnavailableError as exc:
                raise SystemExit(str(exc)) from exc

        rows.append(
            {
                "question": case["question"],
                "expected_page": case["expected_page"],
                "top_page": retrieved_pages[0],
                "hit_at_1": int(retrieved_pages[0] == case["expected_page"]),
                "hit_at_3": int(case["expected_page"] in retrieved_pages[:3]),
                "hit_at_5": int(case["expected_page"] in retrieved_pages),
                "top_score": round(results[0].score, 4),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": (
                    round(generation_ms, 2) if generation_ms is not None else ""
                ),
                "grounded": grounded if grounded is not None else "",
                "citation_valid": (
                    citation_valid if citation_valid is not None else ""
                ),
                "answer": answer_text,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    print(f"Wrote {count} evaluation rows to {args.output}")
    for metric in ("hit_at_1", "hit_at_3", "hit_at_5"):
        value = sum(row[metric] for row in rows) / count
        print(f"{metric}: {value:.1%}")


if __name__ == "__main__":
    main()
