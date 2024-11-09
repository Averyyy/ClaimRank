from tqdm import tqdm
import gc
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import os
# 强制使用 CPU 并禁用并行性
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


print("Loading data...")
df = pd.read_csv('dataset/claims_20241108_100331.csv')
documents = df['claim'].tolist()
print(f"Total documents: {len(documents)}")

dimensions = 512
batch_size = 8  # 使用很小的 batch size

print("Loading model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",
                            truncate_dim=dimensions,
                            device='cpu')  # 强制使用 CPU

print("Starting encoding...")
embeddings = []

try:
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]

        # 确保清理内存
        gc.collect()

        # 编码当前批次
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        embeddings.extend(batch_embeddings)

        # 每处理100个样本保存一次
        if i > 0 and i % 100 == 0:
            print(f"\nSaving checkpoint at {i} documents...")
            temp_embeddings = np.array(embeddings).astype('float32')
            np.save(f'./dataset/embeddings_checkpoint_{i}.npy', temp_embeddings)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    # 保存已处理的结果
    if embeddings:
        temp_embeddings = np.array(embeddings).astype('float32')
        np.save('./dataset/embeddings_last_checkpoint.npy', temp_embeddings)
    raise e

print("Converting to numpy array...")
embeddings = np.array(embeddings).astype('float32')

print("Creating FAISS index...")
index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)

print("Saving index...")
faiss.write_index(index, "./dataset/claim_vector.index")

print("Done!")
