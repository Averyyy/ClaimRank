import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
FILTERED_DATA_PATH = os.path.join(DATA_DIR, 'Filtered_data.csv')
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, 'vector.index')
DOC_SCORES_PATH = os.path.join(DATA_DIR, 'final_document_scores.csv')
TEST_QUERIES_PATH = os.path.join(DATA_DIR, 'test_queries.csv')
TEST_ANSWERS_PATH = os.path.join(DATA_DIR, 'test_answers.csv')

RETRIEVAL_PROMPT_PATH = os.path.join(BASE_DIR, 'prompt', 'retrieval_prompt.txt')
ANSWER_PROMPT_PATH = os.path.join(BASE_DIR, 'prompt', 'answer_prompt.txt')

# Ollama settings
OLLAMA_BASE_URL = 'http://localhost:11434'
# MODEL_NAME = 'gemma2'
MODEL_NAME = 'phi3:medium-128k'

# Retrieval settings
TOP_K = 5  # Number of documents to retrieve
