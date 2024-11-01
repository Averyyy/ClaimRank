import faiss
import numpy as np
import pandas as pd

index = faiss.read_index("./dataset/vector.index")

total_documents = index.ntotal
k = total_documents

similarity_results = []

for idx in range(total_documents):
    embedding = index.reconstruct(idx)
    embedding = np.array(embedding).reshape(1, -1)

    distances, indices = index.search(embedding, k)

    for i in range(1, k):
        id1 = f"{idx:04}"
        id2 = f"{indices[0][i]:04}"
        similarity = 1 / (1 + distances[0][i])

        similarity_results.append((id1, id2, similarity))

similarity_df = pd.DataFrame(similarity_results, columns=['id1', 'id2', 'similarity'])

similarity_df.to_csv('./dataset/similarity_results.csv', index=False)

print(similarity_df.head())
