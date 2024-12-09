import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FILTERED_DATA_PATH = os.path.join(DATA_DIR, 'Filtered_data.csv')
VECTOR_INDEX_PATH = os.path.join(DATA_DIR, 'vector.index')

# Parameters
dimensions = 512  # According to the previous code
model_name = "mixedbread-ai/mxbai-embed-large-v1"


def main():
    if not os.path.exists(FILTERED_DATA_PATH):
        raise FileNotFoundError(f"Filtered_data.csv not found at {FILTERED_DATA_PATH}")
    print(f"Filtered data found at {FILTERED_DATA_PATH}")
    df = pd.read_csv(FILTERED_DATA_PATH, encoding='ISO-8859-1', dtype={'id': str})
    documents = df['text'].tolist()

    # Load model
    model = SentenceTransformer(model_name, device='cpu')
    print(f"Model loaded: {model_name}")

    # Compute embeddings
    print("Computing embeddings...")
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    print(f"Embeddings computed: {embeddings.shape}")

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings)
    print("Index built successfully.")

    # Save the index
    print(f"Saving index to {VECTOR_INDEX_PATH}...")
    faiss.write_index(index, VECTOR_INDEX_PATH)
    print("Index built and saved successfully.")


if __name__ == "__main__":
    main()
