# llama.py
import os
import csv
import time
import logging
import requests
from datetime import datetime
from typing import List, Tuple, Dict

# =================== Hyperparameters ====================

# Paths to input files
FILTERED_CSV_PATH = 'dataset/Filtered_data.csv'
# Update the path to the claims similarity file
CLAIMS_SIMILARITY_CSV_PATH = 'dataset/filtered_claim_similarity_1002-17110.csv'
FILE_ENCODING = 'ISO-8859-1'
SAVE_FILE_ENCODING = 'utf-8'

# Output directory
OUTPUT_DIR = 'dataset'

# Ollama API settings
OLLAMA_BASE_URL = 'http://localhost:11434'
MODEL_NAME = 'gemma2'

# Similarity threshold for comparing claims
SIMILARITY_THRESHOLD = 0.000  # Adjust as needed, now all pairs in CLAIMS_SIMILARITY_CSV_PATH are considered

# Batch sizes
EXTRACTION_BATCH_SIZE = 10  # Number of documents to process in each extraction batch
COMPARISON_BATCH_SIZE = 10  # Number of claim pairs to process in each comparison batch

# Retry settings
MAX_RETRIES = 3

# Logging settings
LOGGING_LEVEL = logging.INFO
LOG_FILE = 'processing.log'

# ========================================================


class OllamaClient:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL

        # Set up logging
        logging.basicConfig(
            level=LOGGING_LEVEL,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load prompts
        with open('llm/prompt/extract.txt', 'r', encoding=SAVE_FILE_ENCODING) as f:
            self.extract_prompt = f.read()
        with open('llm/prompt/compare.txt', 'r', encoding=SAVE_FILE_ENCODING) as f:
            self.compare_prompt = f.read()

    def generate(self, prompt, stream=False, max_retries=MAX_RETRIES):
        """Generate function with retry mechanism."""
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": stream,
                    "max_tokens": 1000,
                    "temperature": 0.0,
                }
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def extract_claims(self, doc_id: str, title: str, text: str, date: str) -> List[Tuple
                                                                                    [str, str, str, str, str, str]]:
        """Extract claims from text with additional context.

        Returns:
            List of tuples containing (claim_id, claim, doc_id, title, text, date)
        """
        try:
            # Include title and date in the prompt
            context = f"Title: {title}\nDate: {date}\nText: {text}"
            prompt = self.extract_prompt.format(text=context)
            response = self.generate(prompt)
            # print('---------' + '\n' + prompt + '\n' + '---------')
            # print('---------' + '\n' + response['response'] + '\n' + '---------')
            claims = []
            for line in response['response'].split('\n'):
                if line.strip() and line.lstrip().startswith(tuple('0123456789')):
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        claim = parts[1].strip()
                        if len(claim) > 10:  # Basic validation
                            claim_id = f"{doc_id.zfill(4)}{str(len(claims)+1).zfill(4)}"
                            claims.append((claim_id, claim, doc_id, title, text, date))
            if not claims:
                self.logger.warning(f"No valid claims extracted from document {doc_id}")
                # retry until max_retries
                return self.extract_claims(doc_id, title, text, date)

            return claims
        except Exception as e:
            self.logger.error(f"Failed to extract claims from document {doc_id}: {str(e)}")
            return []

    def compare_claims(self, claim1_data: Tuple[str, str, str, str, str, str],
                       claim2_data: Tuple[str, str, str, str, str, str]) -> Tuple[str, str, int, str]:
        """Compare two claims and return the result."""
        claim1_id, claim1_text, _, title, text, date = claim1_data
        claim2_id, claim2_text, _, title, text, date = claim2_data
        try:
            prompt = self.compare_prompt.format(claim1=claim1_text, claim2=claim2_text)
            response = self.generate(prompt)
            result = response['response']

            output_lines = [line.strip() for line in result.split('\n') if line.strip().startswith('Output:')]
            if not output_lines:
                raise ValueError(f"No Output section found in response:\n{result}")

            output_line = output_lines[0]
            result_number = output_line.replace('Output:', '').strip()
            result_number = ''.join(filter(lambda x: x in '-0123456789', result_number))

            if result_number in ['1', '-1', '0']:
                return claim1_id, claim2_id, int(result_number), result.replace('\n', ' ').replace('\r', ' ').strip()
            else:
                raise ValueError(f"Invalid comparison result: {result_number} in response:\n{result}")
        except Exception as e:
            self.logger.error(f"Failed to compare claims {claim1_id} and {claim2_id}: {str(e)}")
            return claim1_id, claim2_id, 0, str(e)


def process_documents():
    client = OllamaClient()
    claims_data = []
    relations_data = []

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output files
    claims_file = os.path.join(OUTPUT_DIR, f'claims_{timestamp}.csv')
    relations_file = os.path.join(OUTPUT_DIR, f'relations_{timestamp}.csv')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for existing claims file
    existing_claims_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('claims_') and f.endswith('.csv')]

    if existing_claims_files:
        # Use the most recent claims file
        latest_claims_file = max(existing_claims_files)
        client.logger.info(f"Found existing claims file: {latest_claims_file}")

        with open(os.path.join(OUTPUT_DIR, latest_claims_file), 'r', encoding=SAVE_FILE_ENCODING) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            claims_data = [tuple(row) for row in reader]

        client.logger.info(f"Loaded {len(claims_data)} existing claims")
    else:
        # Load documents from filtered.csv
        client.logger.info(f"Loading documents from {FILTERED_CSV_PATH}")
        documents = []
        with open(FILTERED_CSV_PATH, 'r', encoding=FILE_ENCODING) as f:
            reader = csv.DictReader(f)
            for row in reader:
                documents.append({
                    'id': row['id'],
                    'title': row['title'],
                    'text': row['text'],
                    'date': row.get('date', ''),  # Add date field
                    'validity': row['validity']
                })

        # Extract claims in batches
        client.logger.info(f"Extracting claims from {len(documents)} documents")

        for i in range(0, len(documents), EXTRACTION_BATCH_SIZE):
            batch_docs = documents[i:i + EXTRACTION_BATCH_SIZE]
            for doc in batch_docs:
                doc_id = doc['id']
                title = doc['title']
                text = doc['text']
                date = doc['date']
                claims = client.extract_claims(doc_id, title, text, date)
                claims_data.extend(claims)

            # Save extracted claims after each batch
            with open(claims_file, 'w', newline='', encoding=SAVE_FILE_ENCODING) as f:
                writer = csv.writer(f)
                writer.writerow(['claim_id', 'claim', 'document_id', 'title', 'text', 'date'])
                writer.writerows(claims_data)
            client.logger.info(f"Extracted claims from {i + len(batch_docs)}/{len(documents)} documents")

    # Build a mapping from claim IDs to claim data
    claim_id_to_data: Dict[str, Tuple[str, str, str]] = {}
    for claim in claims_data:
        claim_id, claim_text, doc_id, title, text, date = claim
        claim_id_to_data[claim_id] = claim

    # Load claim similarity data
    client.logger.info(f"Loading claim similarity data from {CLAIMS_SIMILARITY_CSV_PATH}")
    similarity_pairs = []
    with open(CLAIMS_SIMILARITY_CSV_PATH, 'r', encoding=FILE_ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            similarity = float(row['similarity'])
            if similarity >= SIMILARITY_THRESHOLD:
                similarity_pairs.append((row['id1'], row['id2'], similarity))

    client.logger.info(f"Total similar claim pairs above threshold {SIMILARITY_THRESHOLD}: {len(similarity_pairs)}")
    # Create empty relations file immediately
    with open(relations_file, 'w', newline='', encoding=SAVE_FILE_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2', 'relation', 'response'])

    # Prepare claim pairs for comparison based on claim similarities
    processed_pairs = set()

    for idx in range(0, len(similarity_pairs), COMPARISON_BATCH_SIZE):
        batch_pairs = similarity_pairs[idx: idx + COMPARISON_BATCH_SIZE]
        for pair in batch_pairs:
            claim1_id = pair[0]
            claim2_id = pair[1]
            # similarity = float(pair[2])  # Similarity score, you can use it if needed

            # Check if claim IDs exist in the claims data
            claim1_data = claim_id_to_data.get(claim1_id)
            claim2_data = claim_id_to_data.get(claim2_id)
            if not claim1_data or not claim2_data:
                # Skip if claim data not found
                client.logger.warning(f"Claim data not found for IDs: {claim1_id}, {claim2_id}")
                continue

            claim_pair_key = (claim1_id, claim2_id)
            if claim_pair_key in processed_pairs:
                continue
            processed_pairs.add(claim_pair_key)

            res = client.compare_claims(claim1_data, claim2_data)
            if res[2] != 0:
                relations_data.append(res)

        # Save intermediate relations
        with open(relations_file, 'a', newline='', encoding=SAVE_FILE_ENCODING) as f:
            writer = csv.writer(f)
            writer.writerows(relations_data)
        relations_data.clear()
        # Optionally, sleep between batches to control processing speed
        time.sleep(0.1)

        if idx % (COMPARISON_BATCH_SIZE * 10) == 0:
            client.logger.info(f"Processed {idx}/{len(similarity_pairs)} similar claim pairs")

    client.logger.info(f"Comparison complete.")
    client.logger.info(f"Processing complete. Final results saved to {relations_file}")


def main():
    process_documents()


if __name__ == "__main__":
    main()
