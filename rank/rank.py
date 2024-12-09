import os
import pandas as pd
import numpy as np


def claim_relations_to_doc_relations(claim_relations_df):
    claim_relations_df['doc_id1'] = claim_relations_df['id1'].str[:4]
    claim_relations_df['doc_id2'] = claim_relations_df['id2'].str[:4]

    doc_relation_counts = claim_relations_df.groupby(['doc_id1', 'doc_id2'])['relation'].agg(
        support_count=lambda x: (x == 1).sum(),
        oppose_count=lambda x: (x == -1).sum()
    ).reset_index()

    doc_relation_counts['net_relation'] = doc_relation_counts['support_count'] - doc_relation_counts['oppose_count']
    max_net_relation = doc_relation_counts['net_relation'].abs().max()
    if max_net_relation == 0:
        doc_relation_counts['normalized_relation'] = 0
    else:
        doc_relation_counts['normalized_relation'] = doc_relation_counts['net_relation'] / max_net_relation

    document_relations_df = doc_relation_counts[['doc_id1', 'doc_id2', 'normalized_relation']]

    return document_relations_df


def trustrank_core(relations_df, s_seed, max_iterations=100, alpha=0.2, tolerance=1e-6):
    """
    使用给定的种子分布s_seed计算TrustRank或Anti-TrustRank。
    s = alpha * f_I + (1 - alpha) * s_seed_vec
    其中f_I = (I + 1)/2, I = P_pos - N_neg

    若s_seed为空（无种子），则返回一个全0向量。
    """

    # 提取所有的文档ID
    document_ids = set(relations_df['doc_id1']).union(set(relations_df['doc_id2']))
    document_ids = sorted(document_ids)
    doc_index = {doc_id: idx for idx, doc_id in enumerate(document_ids)}
    index_doc = {idx: doc_id for doc_id, idx in doc_index.items()}
    N = len(document_ids)

    # 若无种子，则返回全0.5向量
    if len(s_seed) == 0:
        return {d_id: 0.5 for d_id in document_ids}

    # 将 s_seed 转化为向量形式
    s_seed_vec = np.zeros(N)
    for d_id, val in s_seed.items():
        if d_id in doc_index:
            s_seed_vec[doc_index[d_id]] = val

    # 归一化 s_seed_vec
    sum_seed = s_seed_vec.sum()
    if sum_seed > 0:
        s_seed_vec = s_seed_vec / sum_seed
    else:
        # 无法归一化时，返回全0
        return {d_id: 0.0 for d_id in document_ids}

    # 构建正负向邻接矩阵
    W_plus = np.zeros((N, N))
    W_minus = np.zeros((N, N))
    for _, row in relations_df.iterrows():
        doc_id1 = row['doc_id1']
        doc_id2 = row['doc_id2']
        normalized_relation = row['normalized_relation']
        i = doc_index[doc_id1]
        j = doc_index[doc_id2]
        if normalized_relation > 0:
            W_plus[i, j] += normalized_relation
        elif normalized_relation < 0:
            W_minus[i, j] -= normalized_relation

    W_plus_sum = W_plus.sum(axis=0)
    W_minus_sum = W_minus.sum(axis=0)

    # 初始化s向量为0，为体现仅由迭代和种子决定
    s = np.zeros(N)

    for iteration in range(max_iterations):
        s_prev = s.copy()
        P_pos = np.zeros(N)
        N_neg = np.zeros(N)

        # 正负向影响
        for d in range(N):
            if W_plus_sum[d] > 0:
                P_pos[d] = np.dot(W_plus[:, d], s_prev) / W_plus_sum[d]
            else:
                P_pos[d] = 0

            if W_minus_sum[d] > 0:
                N_neg[d] = np.dot(W_minus[:, d], s_prev) / W_minus_sum[d]
            else:
                N_neg[d] = 0

        I = P_pos - N_neg
        f_I = (I + 1) / 2

        s = alpha * f_I + (1 - alpha) * s_seed_vec

        delta = np.linalg.norm(s - s_prev, ord=1)
        if delta < tolerance:
            print(f'Converged after {iteration + 1} iterations.')
            break
    else:
        print(f'Max iterations reached: {max_iterations}')

    scores = {index_doc[idx]: s[idx] for idx in range(N)}
    return scores


def normalize_scores(scores):
    """
    Normalizes the scores to a range between 0 and 1.

    Parameters:
        scores (dict): Dictionary of document scores.

    Returns:
        dict: Dictionary of normalized document scores.
    """
    min_score = min(scores.values())
    max_score = max(scores.values())
    range_score = max_score - min_score

    if range_score == 0:
        # All scores are the same
        return {k: 0.5 for k in scores}

    normalized = {k: (v - min_score) / range_score for k, v in scores.items()}
    return normalized


if __name__ == "__main__":
    # 根据脚本位置获取数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'dataset')

    relations_path = os.path.join(data_dir, 'relations_latest.csv')
    filtered_data_path = os.path.join(data_dir, 'Filtered_data.csv')

    df = pd.read_csv(relations_path, usecols=['id1', 'id2', 'relation'], dtype={'id1': str, 'id2': str})
    document_relations_df = claim_relations_to_doc_relations(df)

    filtered_data_df = pd.read_csv(filtered_data_path, dtype={'id': str}, encoding='latin1')

    # 获取 top 200 文档
    doc_link_counts_1 = document_relations_df.groupby('doc_id1').size().reset_index(name='count1')
    doc_link_counts_2 = document_relations_df.groupby('doc_id2').size().reset_index(name='count2')
    doc_link_counts = pd.merge(doc_link_counts_1, doc_link_counts_2, left_on='doc_id1', right_on='doc_id2', how='outer')
    doc_link_counts['doc_id'] = doc_link_counts['doc_id1'].combine_first(doc_link_counts['doc_id2'])
    doc_link_counts['count1'] = doc_link_counts['count1'].fillna(0)
    doc_link_counts['count2'] = doc_link_counts['count2'].fillna(0)
    doc_link_counts['total_links'] = doc_link_counts['count1'] + doc_link_counts['count2']
    doc_link_counts = doc_link_counts[['doc_id', 'total_links']].dropna()
    doc_link_counts = doc_link_counts.sort_values('total_links', ascending=False)

    top_docs = doc_link_counts.head(200)['doc_id'].tolist()

    # ground truth
    ground_truth_map = filtered_data_df.set_index('id')['validity'].to_dict()

    # 构建可信种子和不可信种子分布
    trust_seed = {}
    anti_trust_seed = {}
    for doc_id in top_docs:
        validity = ground_truth_map.get(doc_id, None)
        if validity == 1:
            trust_seed[doc_id] = 1.0
        elif validity == 0:
            anti_trust_seed[doc_id] = 1.0

    # 计算TrustRank和Anti-TrustRank分数
    trust_scores = trustrank_core(document_relations_df, s_seed=trust_seed, alpha=0.5)
    anti_trust_scores = trustrank_core(document_relations_df, s_seed=anti_trust_seed, alpha=0.5)

    # 计算最终分数
    final_scores = {}
    all_docs = set(trust_scores.keys()).union(set(anti_trust_scores.keys()))
    for d_id in all_docs:
        t_score = trust_scores.get(d_id, 0.0)
        a_score = anti_trust_scores.get(d_id, 0.0)
        final_scores[d_id] = t_score - a_score

    # 可选放大，以便更直观观察区分度
    scale_factor = 1000
    scaled_final_scores = {d_id: score * scale_factor for d_id, score in final_scores.items()}

    # 输出原始放大分数
    # print(f"Total documents scored: {len(scaled_final_scores)}")
    # for doc_id, score in sorted(scaled_final_scores.items(), key=lambda x: x[0]):
    #     print(f"Document {doc_id} 's score: {score:.4f}")

    # 添加归一化步骤
    normalized_final_scores = normalize_scores(scaled_final_scores)

    # 输出归一化分数
    print("\nNormalized Scores:")
    for doc_id, score in sorted(normalized_final_scores.items(), key=lambda x: x[0]):
        print(f"Document {doc_id} 's normalized score: {score:.4f}")

    # 保存结果到CSV文件
    results_df = pd.DataFrame({
        'doc_id': sorted(all_docs),
        'scaled_score': [scaled_final_scores.get(doc_id, 0.0) for doc_id in sorted(all_docs)],
        'normalized_score': [normalized_final_scores.get(doc_id, 0.0) for doc_id in sorted(all_docs)]
    })

    output_csv_path = os.path.join(current_dir, 'final_document_scores.csv')
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nFinal scores have been saved to {output_csv_path}")
