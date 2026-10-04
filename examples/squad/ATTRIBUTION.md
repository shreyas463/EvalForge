# SQuAD 1.1 derived retrieval subset

Source: Stanford Question Answering Dataset, Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang (2016), based on Wikipedia passages and crowdworker questions/answers. https://rajpurkar.github.io/SQuAD-explorer/

The cases and document excerpts in this directory are adapted from the official development split and distributed under **CC BY-SA 4.0**: https://creativecommons.org/licenses/by-sa/4.0/
Wikipedia article titles, source links in manifest.json, and upstream question IDs are retained for attribution.

Changes: deterministic paragraph subset; passages converted to Markdown; labels converted to JSONL; candidate source files deliberately removed. No model outputs were fabricated. EvalForge source code retains its separate repository license.
