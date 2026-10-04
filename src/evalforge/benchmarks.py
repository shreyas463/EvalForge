"""Reproducible, attributed SQuAD conversion; never send answer labels to targets."""

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from evalforge.datasets import load_jsonl, parse_json
from evalforge.models import EvalCase
from evalforge.storage import write_json

SQUAD_REVISION = "eee5fdbf62f8613a7812b03419e6b29617b74fd1"
SQUAD_URL = (
    f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_REVISION}"
    "/dataset/dev-v1.1.json"
)
SQUAD_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"


def import_squad(source, destination, *, paragraphs=100, questions_per_paragraph=2):
    """Convert pinned dev-v1.1 data, sampling by content ID, without overwriting files."""
    if not 1 <= paragraphs <= 2000 or not 1 <= questions_per_paragraph <= 20:
        raise ValueError("paragraphs must be 1..2000 and questions per paragraph 1..20")
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SQUAD_SHA256:
        raise ValueError("SQuAD source checksum differs from the pinned official development split")
    data = parse_json(raw.decode("utf-8"))
    pool = []
    for article_index, article in enumerate(data["data"]):
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            document = f"article-{article_index:02d}-paragraph-{paragraph_index:03d}.md"
            pool.append(
                (hashlib.sha256(document.encode()).hexdigest(), document, article, paragraph)
            )
    selected = sorted(pool)[:paragraphs]
    cases, documents = [], {}
    for _, document, article, paragraph in selected:
        documents[document] = f"# {article['title'].replace('_', ' ')}\n\n{paragraph['context']}\n"
        for qa in sorted(paragraph["qas"], key=lambda q: q["id"])[:questions_per_paragraph]:
            answers = list(dict.fromkeys(answer["text"] for answer in qa["answers"]))
            for answer in qa["answers"]:
                start, text = answer["answer_start"], answer["text"]
                if paragraph["context"][start : start + len(text)] != text:
                    raise ValueError("source answer span does not match its context")
            cases.append(
                EvalCase(
                    id=qa["id"],
                    input=qa["question"],
                    reference_answer=answers[0],
                    category=article["title"],
                    metadata={
                        "relevant_document_ids": [document],
                        "reference_answers": answers,
                        "source": "SQuAD 1.1",
                        "split": "dev",
                        "wikipedia_article": "https://en.wikipedia.org/wiki/"
                        + quote(article["title"], safe=""),
                    },
                )
            )
    destination = Path(destination)
    if destination.exists():
        raise ValueError("destination already exists; choose a new directory")
    destination.mkdir(parents=True)
    missing = sorted(documents)[: max(1, len(documents) // 10)]
    for label in ("baseline", "candidate"):
        folder = destination / "documents" / label
        folder.mkdir(parents=True)
        for document, text in documents.items():
            if label == "baseline" or document not in missing:
                (folder / document).write_text(text, encoding="utf-8")
    (destination / "cases.jsonl").write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    load_jsonl(destination / "cases.jsonl")
    config = {
        "dataset": "cases.jsonl",
        "dataset_name": "squad-dev-retrieval-subset",
        "baseline": {
            "kind": "retrieval",
            "name": "complete-corpus",
            "documents": "documents/baseline",
            "top_k": 5,
        },
        "candidate": {
            "kind": "retrieval",
            "name": "missing-source-documents",
            "documents": "documents/candidate",
            "top_k": 5,
        },
        "evaluators": [
            {"metric": "source_recall", "kind": "retrieval_recall", "options": {"k": 5}}
        ],
        "gates": [{"metric": "source_recall", "max_regression": 0}],
        "execution": {"concurrency": 2, "max_seconds": 120},
        "output_dir": "../../.evalforge",
    }
    write_json(destination / "retrieval.json", config)
    passing = json.loads(json.dumps(config))
    passing["candidate"] = {**passing["baseline"], "name": "unchanged-corpus"}
    write_json(destination / "unchanged.json", passing)
    (destination / "ATTRIBUTION.md").write_text(
        "# SQuAD 1.1 derived retrieval subset\n\n"
        "Source: Stanford Question Answering Dataset, Pranav Rajpurkar, Jian Zhang, "
        "Konstantin Lopyrev and Percy Liang (2016), based on Wikipedia passages and "
        "crowdworker questions/answers. https://rajpurkar.github.io/SQuAD-explorer/\n\n"
        "The cases and document excerpts in this directory are adapted from the official "
        "development split and distributed under **CC BY-SA 4.0**: "
        "https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Wikipedia article titles, source links in manifest.json, and upstream question IDs "
        "are retained for attribution.\n\n"
        "Changes: deterministic paragraph subset; passages converted to Markdown; labels "
        "converted to JSONL; candidate source files deliberately removed. No model outputs "
        "were fabricated. EvalForge source code retains its separate repository license.\n",
        encoding="utf-8",
    )
    write_json(
        destination / "manifest.json",
        {
            "source_url": SQUAD_URL,
            "source_revision": SQUAD_REVISION,
            "source_sha256": SQUAD_SHA256,
            "license": "CC-BY-SA-4.0",
            "split": "dev",
            "selection": "SHA-256 of stable paragraph IDs; ascending; first N",
            "paragraphs": len(documents),
            "questions": len(cases),
            "questions_per_paragraph": questions_per_paragraph,
            "articles": len({case.category for case in cases}),
            "document_sources": {
                document: {
                    "title": article["title"],
                    "wikipedia_article": "https://en.wikipedia.org/wiki/"
                    + quote(article["title"], safe=""),
                }
                for _, document, article, _ in selected
            },
            "removed_documents": missing,
            "files": {
                str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob("*"))
                if path.is_file()
            },
        },
    )
    return {"questions": len(cases), "documents": len(documents), "destination": str(destination)}
