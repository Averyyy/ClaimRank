import faiss
import numpy as np
import pandas as pd
from .config import VECTOR_INDEX_PATH, FILTERED_DATA_PATH, TOP_K


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(VECTOR_INDEX_PATH)
        self.docs_df = pd.read_csv(FILTERED_DATA_PATH, encoding='ISO-8859-1', dtype={'id': str})
        self.docs_df['id'] = self.docs_df['id'].str.zfill(4)

    def retrieve(self, query_embedding: np.ndarray):
        # query_embedding: numpy array of shape (1, D)
        distances, indices = self.index.search(query_embedding, TOP_K)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc_id = f"{idx:04}"
            doc_record = self.docs_df[self.docs_df['id'] == doc_id].iloc[0]
            results.append({
                'doc_id': doc_id,
                'title': doc_record['title'],
                'text': doc_record['text'],
                'distance': float(dist)
            })
        return results

    def get_embedding(self, model, query: str) -> np.ndarray:
        # model: a sentence embedding model (like sentence-transformers)
        emb = model.encode([query])
        emb = np.array(emb).astype('float32')
        return emb
