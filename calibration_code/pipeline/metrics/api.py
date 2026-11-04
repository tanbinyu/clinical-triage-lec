"""Stable callable API for computing and writing a metrics bundle."""
import json
from pathlib import Path

from pipeline.metrics.run_all import compute_all_metrics, generate_markdown


def compute_and_write_metrics(beta, alpha, data_root, input_root, output_dir,
                              stem="metrics"):
    """Compute metrics and write UTF-8 JSON/Markdown without module globals."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = compute_all_metrics(
        beta=beta,
        alpha=alpha,
        data_root=data_root,
        input_root=input_root,
    )

    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown(results))
    return results, json_path, markdown_path
