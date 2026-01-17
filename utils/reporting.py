from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np


def save_markdown_report(
    out_dir: str | Path,
    title: str,
    dataset_info: dict,
    embedding_info: str,
    results,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = out_dir / f"training_report_{ts}.md"

    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Dataset Info\n")
    for k, v in dataset_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Embedding\n")
    lines.append(f"- {embedding_info}\n")

    lines.append("## Model Results\n")
    for r in results:
        lines.append(f"### {r.name.upper()}  (Accuracy: {r.accuracy:.4f})\n")
        lines.append("**Classification Report**\n")
        lines.append("```text")
        lines.append(r.report.strip())
        lines.append("```\n")
        lines.append("**Confusion Matrix**\n")
        lines.append("```text")
        lines.append(np.array2string(r.cm))
        lines.append("```\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
