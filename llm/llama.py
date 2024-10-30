# llama.py
import os
import json
import csv
import time
import logging
from typing import List
import requests
from datetime import datetime


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load prompts
        with open('llm/prompt/extract.txt', 'r') as f:
            self.extract_prompt = f.read()
        with open('llm/prompt/compare.txt', 'r') as f:
            self.compare_prompt = f.read()

    def generate(self, prompt, model="llama3.2", stream=False, max_retries=3):
        """Generate response with retry mechanism"""
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": stream
                }
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def extract_claims(self, text: str, max_retries=3) -> List[str]:
        """Extract claims from text with validation"""
        for attempt in range(max_retries):
            try:
                prompt = self.extract_prompt.format(text=text)
                response = self.generate(prompt)

                claims = []
                for line in response['response'].split('\n'):
                    if line.strip() and line.lstrip().startswith(tuple('0123456789')):
                        parts = line.split('.', 1)
                        if len(parts) == 2:
                            claim = parts[1].strip()
                            if len(claim) > 10:  # Basic validation - ensure claim is substantial
                                claims.append(claim)

                if not claims:
                    raise ValueError("No valid claims extracted")

                return claims
            except Exception as e:
                self.logger.warning(f"Claim extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to extract claims from text: {text[:100]}...")
                    return []
                time.sleep(2 ** attempt)

    def compare_claims(self, claim1: str, claim2: str, max_retries=3) -> int:
        """Compare two claims with validation"""
        for attempt in range(max_retries):
            try:
                prompt = self.compare_prompt.format(claim1=claim1, claim2=claim2)
                response = self.generate(prompt)
                result = response['response']
                print(result)
                output_lines = [line.strip() for line in result.split('\n') if line.strip().startswith('Output:')]
                if not output_lines:
                    raise ValueError("No Output section found")

                output_line = output_lines[0]
                result_number = output_line.replace('Output:', '').strip()
                result_number = ''.join(filter(lambda x: x in '-0123456789', result_number))

                if result_number in ['1', '-1', '0']:
                    return int(result_number)
                else:
                    raise ValueError(f"Invalid comparison result: {result_number}")

            except Exception as e:
                self.logger.warning(f"Claim comparison attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to compare claims:\nClaim1: {claim1}\nClaim2: {claim2}")
                    return 0
                time.sleep(2 ** attempt)


def process_documents(input_dir: str, output_dir: str):
    client = OllamaClient()
    claims_data = []
    relations_data = []

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for existing claims file
    existing_claims_files = [f for f in os.listdir(output_dir) if f.startswith('claims_') and f.endswith('.csv')]

    if existing_claims_files:
        # Use the most recent claims file
        latest_claims_file = max(existing_claims_files)
        client.logger.info(f"Found existing claims file: {latest_claims_file}")

        with open(os.path.join(output_dir, latest_claims_file), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            claims_data = list(reader)

        client.logger.info(f"Loaded {len(claims_data)} existing claims")
    else:
        # Process each JSON file to extract claims
        total_files = len([f for f in os.listdir(input_dir) if f.endswith('.json')])
        file_count = 0

        for filename in os.listdir(input_dir):
            if not filename.endswith('.json'):
                continue

            file_count += 1
            client.logger.info(f"Processing file {file_count}/{total_files}: {filename}")

            try:
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                    documents = json.load(f)

                for doc in documents:
                    doc_id = str(doc['id'])
                    client.logger.info(f"Processing document ID: {doc_id}")

                    claims = client.extract_claims(doc['text'])
                    client.logger.info(f"Extracted {len(claims)} claims from document {doc_id}")

                    # Store claims with IDs
                    for i, claim in enumerate(claims, 1):
                        # Format claim_id as string concatenation
                        claim_id = f"{doc_id.zfill(8)}{str(i).zfill(8)}"
                        claims_data.append([claim_id, claim, doc_id])

                    # Save intermediate claims after each document
                    claims_file = os.path.join(output_dir, f'claims_{timestamp}.csv')
                    with open(claims_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['claim_id', 'claim', 'document_id'])
                        writer.writerows(claims_data)
            except Exception as e:
                client.logger.error(f"Error processing file {filename}: {str(e)}")
                continue

        client.logger.info(f"Claim extraction complete. Total claims extracted: {len(claims_data)}")

    # Compare claims between different documents
    total_claims = len(claims_data)
    total_comparisons = total_claims * (total_claims - 1) // 2
    comparison_count = 0

    # To avoid duplicate comparisons, use a set to track processed pairs
    processed_pairs = set()

    start_time = time.time()
    for i in range(total_claims):
        claim1_data = claims_data[i]
        claim1_id = claim1_data[0]
        claim1_text = claim1_data[1]
        doc1_id = claim1_data[2]

        for j in range(i + 1, total_claims):
            claim2_data = claims_data[j]
            claim2_id = claim2_data[0]
            claim2_text = claim2_data[1]
            doc2_id = claim2_data[2]

            if doc1_id == doc2_id:
                continue  # Skip claims from the same document

            # Check if this pair has already been processed
            pair_key = (claim1_id, claim2_id)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            relation = client.compare_claims(claim1_text, claim2_text)
            # if relation != 0:  # Only store non-zero relations
            relations_data.append([claim1_id, claim2_id, relation])

            comparison_count += 1
            if comparison_count % 100 == 0 or comparison_count == total_comparisons:
                elapsed_time = time.time() - start_time
                client.logger.info(
                    f"Processed {comparison_count}/{total_comparisons} comparisons, Elapsed time: {elapsed_time:.2f}s")
                # Save intermediate relations every 100 comparisons
                relations_file = os.path.join(output_dir, f'relations_{timestamp}.csv')
                with open(relations_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id1', 'id2', 'relation'])
                    writer.writerows(relations_data)

    client.logger.info(f"Comparison complete. Total relations found: {len(relations_data)}")

    # Save final relations file
    relations_file = os.path.join(output_dir, f'relations_{timestamp}.csv')
    with open(relations_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id1', 'id2', 'relation'])
        writer.writerows(relations_data)

    client.logger.info(f"Processing complete. Final results saved to {relations_file}")
    client.logger.info(f"Processed {len(claims_data)} claims and {len(relations_data)} relations")


if __name__ == "__main__":
    process_documents(
        input_dir='dataset/enwiki20201020',
        output_dir='dataset'
    )
