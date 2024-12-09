import argparse
import sys
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from .retriever import Retriever
from .ranker import Ranker
from .llm_client import LLMClient
from .config import MODEL_NAME, FILTERED_DATA_PATH
from .evaluator import compare_modes
from .utils import load_prompt_template, format_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["vanilla", "score"],
        default="vanilla", help="Retrieval mode: vanilla or score")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate both modes and output metrics")
    parser.add_argument("--llm_evaluate", action="store_true", help="Use LLM to evaluate answers and produce a score")
    parser.add_argument("--query", type=str, default=None, help="Run a single query in specified mode")
    args = parser.parse_args()

    # Load embeddings model
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device='cpu', truncate_dim=1024)

    retriever = Retriever()
    ranker = Ranker()
    llm_client = LLMClient()  # For retrieval and answering
    evaluation_llm_client = LLMClient()  # Separate client instance if needed for evaluation

    if args.evaluate:
        comparison_df, results_df = compare_modes(
            retriever,
            ranker,
            llm_client,
            embedding_model,
            evaluation_llm_client=evaluation_llm_client,
            llm_evaluate=args.llm_evaluate
        )
        comparison_df.to_csv("rag_evaluation_comparison.csv", index=False)
        results_df.to_csv("rag_detailed_results.csv", index=False)
        print("Evaluation complete. Results saved to rag_evaluation_comparison.csv and rag_detailed_results.csv")
        sys.exit(0)

    if args.query is not None:
        query_embedding = retriever.get_embedding(embedding_model, args.query)
        retrieved_docs = retriever.retrieve(query_embedding)
        if args.mode == 'score':
            retrieved_docs = ranker.rank_with_scores(retrieved_docs)

        retrieval_prompt_template = load_prompt_template('./rag/prompt/retrieval_prompt.txt')
        formatted_docs = format_documents(retrieved_docs)
        prompt = retrieval_prompt_template.format(documents=formatted_docs, query=args.query)
        answer = llm_client.generate(prompt)
        print("RAG Answer:\n", answer)
    else:
        print("No query provided and no evaluation requested. Use --query to ask a question or --evaluate to run tests.")


if __name__ == "__main__":
    main()
