ClaimTrust-A-Propagation-Based-Trust-Scoring-Framework-for-Retrieval-Augmented-Generation-Systems

dataset: partition from https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

# Requirements

- Python 3.10 Environment
- Ollama server

# TO RUN EVALUATION

## LLM Evaluation (Recommended)

```bash
python3 -m rag.main --evaluate --llm_evaluate
```

## Basic evaluation:

```bash
python3 -m rag.main --evaluate
```

## Vanilla mode to test a single query:

```bash
python3 -m rag.main --query "What did MIT say about Trump's climate research understanding?" --mode vanilla
```

## score mode to test a single query:

```bash
python3 -m rag.main --query "What did MIT say about Trump's climate research understanding?" --mode score
```
