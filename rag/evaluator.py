import pandas as pd
from tqdm import tqdm
from .config import TEST_QUERIES_PATH, TEST_ANSWERS_PATH
from .utils import load_prompt_template, format_documents

# Simple evaluation metric: exact match or substring check in answer
# You can improve metrics as needed.


def evaluate_system(retriever, ranker, llm_client, embedding_model, mode='vanilla'):
    # mode: 'vanilla' or 'score'

    queries_df = pd.read_csv(TEST_QUERIES_PATH)
    answers_df = pd.read_csv(TEST_ANSWERS_PATH)

    # Map query_id -> expected_answer
    ground_truth = dict(zip(answers_df['query_id'], answers_df['answer'].astype(str)))

    results = []
    retrieval_prompt_template = load_prompt_template('./rag/prompt/retrieval_prompt.txt')

    for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df)):
        query_id = row['query_id']
        query = row['query']

        query_embedding = retriever.get_embedding(embedding_model, query)
        retrieved_docs = retriever.retrieve(query_embedding)

        if mode == 'score':
            retrieved_docs = ranker.rank_with_scores(retrieved_docs)

        # Create a prompt with retrieved documents
        formatted_docs = format_documents(retrieved_docs)
        prompt = retrieval_prompt_template.format(documents=formatted_docs, query=query)

        answer = llm_client.generate(prompt)
        expected = ground_truth.get(query_id, "")

        # Simple metric: check if expected answer is in answer
        correctness = 1 if expected.lower() in answer.lower() else 0

        results.append({
            'query_id': query_id,
            'mode': mode,
            'query': query,
            'expected_answer': expected,
            'model_answer': answer,
            'correct': correctness
        })

    return pd.DataFrame(results)


def compare_modes(retriever, ranker, llm_client, embedding_model):
    df_vanilla = evaluate_system(retriever, ranker, llm_client, embedding_model, mode='vanilla')
    df_score = evaluate_system(retriever, ranker, llm_client, embedding_model, mode='score')

    # Calculate aggregate metrics
    vanilla_accuracy = df_vanilla['correct'].mean()
    score_accuracy = df_score['correct'].mean()

    comparison_df = pd.DataFrame([{
        'method': 'vanilla',
        'accuracy': vanilla_accuracy
    }, {
        'method': 'score',
        'accuracy': score_accuracy
    }])

    # Combine into one CSV
    results_df = pd.concat([df_vanilla, df_score], ignore_index=True)
    return comparison_df, results_df
