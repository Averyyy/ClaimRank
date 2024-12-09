import pandas as pd
from tqdm import tqdm
from .config import TEST_QUERIES_PATH, TEST_ANSWERS_PATH
from .utils import load_prompt_template, format_documents


def evaluate_system(retriever, ranker, llm_client, embedding_model, mode='vanilla'):
    # Existing substring-based evaluation
    queries_df = pd.read_csv(TEST_QUERIES_PATH)
    answers_df = pd.read_csv(TEST_ANSWERS_PATH)

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

        formatted_docs = format_documents(retrieved_docs)
        prompt = retrieval_prompt_template.format(documents=formatted_docs, query=query)

        answer = llm_client.generate(prompt)
        expected = ground_truth.get(query_id, "")

        # Simple substring check metric
        correctness = 1 if expected.lower() in answer.lower() else 0

        results.append({
            'query_id': query_id,
            'mode': mode,
            'query': query,
            'expected_answer': expected,
            'model_answer': answer,
            'correct_substring': correctness
        })

    return pd.DataFrame(results)


def evaluate_system_with_llm(evaluation_llm_client, df_results):
    # LLM-based evaluation to compute a score between 0 and 1 for each response
    # We'll read from df_results that already has the model answer and expected answer.
    evaluation_prompt_template = load_prompt_template('./rag/prompt/evaluation_prompt.txt')
    print(
        "Evaluating model answers using LLM-based scoring. This may take a while depending on the number of queries."
    )

    llm_scores = []
    for idx, row in tqdm(df_results.iterrows(), total=len(df_results)):
        query = row['query']
        expected_answer = row['expected_answer']
        model_answer = row['model_answer']

        eval_prompt = evaluation_prompt_template.format(
            query=query,
            expected_answer=expected_answer,
            model_answer=model_answer
        )
        eval_response = evaluation_llm_client.generate(eval_prompt)

        # Try to parse the response as a float
        try:
            print(eval_response)
            score = float(eval_response.strip())
            if score < 0 or score > 1:
                score = 0.0
        except:
            score = 0.5

        llm_scores.append(score)

    df_results['llm_score'] = llm_scores
    return df_results


def compare_modes(retriever, ranker, llm_client, embedding_model, evaluation_llm_client=None, llm_evaluate=False):
    df_vanilla = evaluate_system(retriever, ranker, llm_client, embedding_model, mode='vanilla')
    df_score = evaluate_system(retriever, ranker, llm_client, embedding_model, mode='score')

    results_df = pd.concat([df_vanilla, df_score], ignore_index=True)

    if llm_evaluate and evaluation_llm_client is not None:
        # Run LLM-based evaluation on the combined results
        results_df = evaluate_system_with_llm(evaluation_llm_client, results_df)

    # Aggregate metrics for substring check
    vanilla_accuracy = df_vanilla['correct_substring'].mean()
    score_accuracy = df_score['correct_substring'].mean()

    # Aggregate metrics for LLM-based scoring if available
    if llm_evaluate and 'llm_score' in results_df.columns:
        vanilla_llm_scores = results_df[results_df['mode'] == 'vanilla']['llm_score'].mean()
        score_llm_scores = results_df[results_df['mode'] == 'score']['llm_score'].mean()
    else:
        vanilla_llm_scores = None
        score_llm_scores = None

    comparison_data = [{
        'method': 'vanilla',
        'substring_accuracy': vanilla_accuracy,
        'llm_avg_score': vanilla_llm_scores
    }, {
        'method': 'score',
        'substring_accuracy': score_accuracy,
        'llm_avg_score': score_llm_scores
    }]
    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df, results_df
