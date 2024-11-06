import faiss
import numpy as np
import pandas as pd

index = faiss.read_index("./dataset/vector.index")

total_documents = index.ntotal
k = total_documents

similarity_results = set()

for idx in range(total_documents):
    embedding = index.reconstruct(idx)
    embedding = np.array(embedding).reshape(1, -1)

    distances, indices = index.search(embedding, k)

    for i in range(idx+1, k):
        id1 = idx
        id2 = indices[0][i]
        pair = tuple(sorted([id1, id2]))

        similarity = 1 / (1 + distances[0][i])
        # similarity = round(similarity, 4)
        similarity_results.add((pair, similarity))

similarity_results = [(f"{pair[0]:04}", f"{pair[1]:04}", similarity) for pair, similarity in similarity_results]
similarity_df = pd.DataFrame(similarity_results, columns=['id1', 'id2', 'similarity'])
similarity_df.to_csv('./dataset/similarity_results.csv', index=False)

print(similarity_df.head())
