import faiss
import numpy as np
import pandas as pd

index = faiss.read_index("./dataset/claim_vector.index")

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

similarity_results = [(pair[0], pair[1], similarity) for pair, similarity in similarity_results]
similarity_df = pd.DataFrame(similarity_results, columns=['id1', 'id2', 'similarity'])


claims_df = pd.read_csv('./dataset/claims_20241108_100331.csv')
id_mapping = {i: claims_df.iloc[i]['claim_id'] for i in range(len(claims_df))}

similarity_df['id1'] = similarity_df['id1'].map(id_mapping)
similarity_df['id2'] = similarity_df['id2'].map(id_mapping)

similarity_df['id1'] = similarity_df['id1'].astype(str).str.zfill(8)
similarity_df['id2'] = similarity_df['id2'].astype(str).str.zfill(8)

df_sorted = similarity_df.sort_values(by=similarity_df.columns[2], ascending=False)
df_first_1000 = df_sorted.head(1000)

df_first_1000.to_csv('./dataset/filtered_claim_similarity_results.csv', index=False)
df_sorted.to_csv('./dataset/sorted_claim_similarity_results.csv', index=False)

print(df_sorted.head())