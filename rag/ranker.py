import pandas as pd
import math
from typing import List, Dict
from .config import DOC_SCORES_PATH


class Ranker:
    def __init__(self):
        self.scores_df = pd.read_csv(DOC_SCORES_PATH, dtype={'doc_id': str})
        self.scores_map = dict(zip(self.scores_df['doc_id'], self.scores_df['normalized_score']))

    def rank_with_scores(self, retrieved_docs: List[Dict]) -> List[Dict]:
        # Re-rank documents by combining normalized_score with similarity
        # Simple approach: final_score = normalized_score / (1 + distance)
        # Or any scoring logic you prefer
        for doc in retrieved_docs:
            doc_score = self.scores_map.get(doc['doc_id'], 0.5)  # default 0.5 if missing
            # Invert distance so lower distance (more similar) = higher score
            final_score = doc_score / (1 + doc['distance'])
            doc['final_score'] = final_score
        # Sort by final_score descending
        ranked = sorted(retrieved_docs, key=lambda x: x['final_score'], reverse=True)
        return ranked
