import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# df = pd.read_csv('./dataset/Filtered_Fake.csv')
df = pd.read_csv('./dataset/Filtered_True.csv')

documents = df['text'].tolist()

dimensions = 512

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')



index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)

# faiss.write_index(index, "./dataset/Fake.index")
faiss.write_index(index, "./dataset/True.index")

