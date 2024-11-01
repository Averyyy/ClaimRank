import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv('./dataset/Filtered_data.csv', encoding='ISO-8859-1')

documents = df['text'].tolist()

dimensions = 512

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')


index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)

faiss.write_index(index, "./dataset/vector.index")
