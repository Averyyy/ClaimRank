# llama.py
import os
import json
import csv
import time
import logging
from typing import List, Tuple
import asyncio
import aiohttp
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

    async def generate(self, session, prompt, model="llama3.2", stream=False, max_retries=3):
        """Asynchronous generate function with retry mechanism."""
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/api/generate"
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": stream
                }
                async with session.post(url, json=payload, timeout=60) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result
            except Exception as e:
                # self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def extract_claims(self, session, doc_id: str, text: str) -> List[Tuple[str, str, str]]:
        """Extract claims from text asynchronously."""
        try:
            prompt = self.extract_prompt.format(text=text)
            response = await self.generate(session, prompt)
            claims = []
            for line in response['response'].split('\n'):
                if line.strip() and line.lstrip().startswith(tuple('0123456789')):
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        claim = parts[1].strip()
                        if len(claim) > 10:  # Basic validation
                            claim_id = f"{doc_id.zfill(8)}{str(len(claims)+1).zfill(8)}"
                            claims.append((claim_id, claim, doc_id))
            if not claims:
                self.logger.warning(f"No valid claims extracted from document {doc_id}")
            return claims
        except Exception as e:
            self.logger.error(f"Failed to extract claims from document {doc_id}: {str(e)}")
            return []

    async def compare_claims(self, session, claim1_data: Tuple[str, str, str],
                             claim2_data: Tuple[str, str, str]) -> Tuple[str, str, int]:
        """Compare two claims asynchronously."""
        claim1_id, claim1_text, _ = claim1_data
        claim2_id, claim2_text, _ = claim2_data
        try:
            prompt = self.compare_prompt.format(claim1=claim1_text, claim2=claim2_text)
            response = await self.generate(session, prompt)
            result = response['response']

            # Extract number from "Output: " section
            output_lines = [line.strip() for line in result.split('\n') if line.strip().startswith('Output:')]
            if not output_lines:
                raise ValueError("No Output section found")

            output_line = output_lines[0]
            result_number = output_line.replace('Output:', '').strip()
            result_number = ''.join(filter(lambda x: x in '-0123456789', result_number))

            if result_number in ['1', '-1', '0']:
                return claim1_id, claim2_id, int(result_number)
            else:
                raise ValueError(f"Invalid comparison result: {result_number}")
        except Exception as e:
            self.logger.error(f"Failed to compare claims {claim1_id} and {claim2_id}: {str(e)}")
            return claim1_id, claim2_id, 0


async def process_documents(input_dir: str, output_dir: str):
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
            claims_data = [tuple(row) for row in reader]

        client.logger.info(f"Loaded {len(claims_data)} existing claims")
    else:
        # Process each JSON file to extract claims asynchronously
        total_files = len([f for f in os.listdir(input_dir) if f.endswith('.json')])
        file_count = 0

        async with aiohttp.ClientSession() as session:
            tasks = []
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
                        text = doc['text']
                        task = asyncio.create_task(
                            client.extract_claims(session, doc_id, text)
                        )
                        tasks.append(task)
                except Exception as e:
                    client.logger.error(f"Error processing file {filename}: {str(e)}")
                    continue

            # Gather all claim extraction tasks
            results = await asyncio.gather(*tasks)
            for claims in results:
                claims_data.extend(claims)

            # Save extracted claims
            claims_file = os.path.join(output_dir, f'claims_{timestamp}.csv')
            with open(claims_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['claim_id', 'claim', 'document_id'])
                writer.writerows(claims_data)
            client.logger.info(f"Claim extraction complete. Total claims extracted: {len(claims_data)}")

    # Compare claims asynchronously
    client.logger.info("Starting claim comparisons...")
    total_claims = len(claims_data)
    comparison_tasks = []
    processed_pairs = set()
    batch_size = 500  # Adjust batch size as needed

    async with aiohttp.ClientSession() as session:
        for i in range(total_claims):
            claim1_data = claims_data[i]
            for j in range(i + 1, total_claims):
                claim2_data = claims_data[j]

                # Skip if from the same document
                if claim1_data[2] == claim2_data[2]:
                    continue

                # Check if this pair has already been processed
                pair_key = (claim1_data[0], claim2_data[0])
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                task = asyncio.create_task(
                    client.compare_claims(session, claim1_data, claim2_data)
                )
                comparison_tasks.append(task)

                # If we've reached the batch size, process the batch
                if len(comparison_tasks) >= batch_size:
                    results = await asyncio.gather(*comparison_tasks)
                    for res in results:
                        if res[2] != 0:  # Only store non-zero relations
                            # if True:
                            relations_data.append(res)
                    # Save intermediate relations
                    relations_file = os.path.join(output_dir, f'relations_{timestamp}.csv')
                    with open(relations_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['id1', 'id2', 'relation'])
                        writer.writerows(relations_data)
                    comparison_tasks = []  # Reset tasks

        # Process any remaining tasks
        if comparison_tasks:
            results = await asyncio.gather(*comparison_tasks)
            for res in results:
                if res[2] != 0:
                    relations_data.append(res)
            # Save final relations
            relations_file = os.path.join(output_dir, f'relations_{timestamp}.csv')
            with open(relations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id1', 'id2', 'relation'])
                writer.writerows(relations_data)

    client.logger.info(f"Comparison complete. Total relations found: {len(relations_data)}")
    client.logger.info(f"Processing complete. Final results saved to {relations_file}")


def main():
    asyncio.run(process_documents(
        input_dir='dataset/enwiki20201020',
        output_dir='dataset'
    ))


if __name__ == "__main__":
    main()
