import requests
import logging
from rag.config import OLLAMA_BASE_URL, MODEL_NAME


class LLMClient:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        logging.basicConfig(level=logging.INFO)

    def generate(self, prompt, stream=False, max_tokens=1000, temperature=0.0):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result['response']
