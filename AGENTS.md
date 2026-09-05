# Repository Guidelines

## Project Structure & Module Organization

This repository implements image captioning with optional RAG retrieval over MS COCO captions. Core Python modules live in `src/`: `encoder.py`, `decoder.py`, `models/`, `dataset.py`, `inference.py`, `metrics.py`, and configuration in `config.py`. Operational scripts are in `script/`, including split preprocessing, vocabulary building, CLIP feature extraction, knowledge-base construction, and RAG context retrieval. Notebooks (`train.ipynb`, `generate_caption.ipynb`, `evaluate.ipynb`, `test.ipynb`) are used as runnable experiment entry points. Generated data is under `artifacts/`; model weights are under `checkpoints/`. Large datasets are configured via `.env` and should stay outside git.

## Build, Test, and Development Commands

Use the shared virtualenv at `/home/tam/Link to workspace/ML/.venv/bin`.

```bash
/home/tam/Link\ to\ workspace/ML/.venv/bin/python script/preprocess_dfs.py
/home/tam/Link\ to\ workspace/ML/.venv/bin/python script/build_vocab.py
/home/tam/Link\ to\ workspace/ML/.venv/bin/python script/build_kb.py
/home/tam/Link\ to\ workspace/ML/.venv/bin/accelerate launch script/retrieve_rag_contexts.py
```

Run preprocessing before vocabulary, KB, or retrieval generation. Rebuild RAG contexts whenever `KB_MODEL_ID`, split parquet files, or `artifacts/kb/*` changes.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and type hints where they clarify interfaces. Keep module names lowercase with underscores, class names in `PascalCase`, and functions/variables in `snake_case`. Prefer `pathlib.Path` for filesystem paths and centralize configurable locations in `src/config.py`. Avoid ad hoc string parsing when pandas, JSON, or parquet APIs are available.

## Testing Guidelines

There is no formal test suite yet. Use focused smoke checks after changes:

```bash
/home/tam/Link\ to\ workspace/ML/.venv/bin/python -m compileall src script
```

For retrieval changes, validate that FAISS row count matches metadata and that known train images retrieve their own captions before filtering.

## Commit & Pull Request Guidelines

History uses concise messages such as `refactor: ...` and imperative descriptions. Keep commits scoped to one behavior or script. PRs should include the problem, changed files, commands run, artifact impacts, and any metric changes from `evaluate.ipynb`.

## Security & Configuration Tips

Do not commit datasets, checkpoints, generated parquet/FAISS files, or secrets. Keep local paths and run mode in `.env`; document any non-default environment variables needed to reproduce an experiment.
