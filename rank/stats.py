import pandas as pd
from collections import defaultdict


def find_clusters(edges):
    """使用并查集找到所有连通分量（聚类）"""
    def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]

    def union(parent, x, y):
        parent[find(parent, x)] = find(parent, y)

    # 初始化并查集
    parent = {}

    # 收集所有唯一的文档ID
    all_docs = set()
    for (doc1, doc2) in edges:
        all_docs.add(doc1)
        all_docs.add(doc2)

    # 初始化每个文档的父节点为自身
    for doc in all_docs:
        parent[doc] = doc

    # 合并有边相连的文档
    for (doc1, doc2) in edges:
        union(parent, doc1, doc2)

    # 找出所有聚类
    clusters = defaultdict(set)
    for doc in all_docs:
        root = find(parent, doc)
        clusters[root].add(doc)

    return dict(clusters)


def count_document_edges(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 创建文档对之间的边计数字典
    edge_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})

    # 收集所有出现的文档ID
    all_docs = set()
    doc_edges = set()  # 用于存储所有有边的文档对

    # 遍历每一行
    for _, row in df.iterrows():
        # 将ID转换为字符串并获取前4位
        doc1 = str(row['id1']).zfill(8)[:4]
        doc2 = str(row['id2']).zfill(8)[:4]

        all_docs.add(doc1)
        all_docs.add(doc2)

        # 确保文档对的顺序一致（较小ID在前）
        doc_pair = tuple(sorted([doc1, doc2]))
        doc_edges.add(doc_pair)

        # 根据relation更新计数
        if row['relation'] == 1:
            edge_counts[doc_pair]['positive'] += 1
        elif row['relation'] == -1:
            edge_counts[doc_pair]['negative'] += 1

    # 找出没有任何边的文档
    docs_with_edges = set()
    for doc1, doc2 in doc_edges:
        docs_with_edges.add(doc1)
        docs_with_edges.add(doc2)

    isolated_docs = all_docs - docs_with_edges

    # 找出所有聚类
    clusters = find_clusters(doc_edges)

    # 创建结果数据框
    results = []
    for doc_pair, counts in edge_counts.items():
        results.append({
            'doc1': doc_pair[0],
            'doc2': doc_pair[1],
            'positive_edges': counts['positive'],
            'negative_edges': counts['negative'],
            'total_edges': counts['positive'] + counts['negative']
        })

    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)

    # 计算总计
    total_positive = results_df['positive_edges'].sum()
    total_negative = results_df['negative_edges'].sum()
    total_edges = results_df['total_edges'].sum()

    # 添加总计行
    totals_row = pd.DataFrame([{
        'doc1': 'TOTAL',
        'doc2': '',
        'positive_edges': total_positive,
        'negative_edges': total_negative,
        'total_edges': total_edges
    }])

    results_df = pd.concat([results_df, totals_row], ignore_index=True)

    # 保存到CSV
    results_df.to_csv(output_file, index=False)

    # 打印统计信息
    print(f"结果已保存到 {output_file}")
    print(f"\n总计:")
    print(f"总正边数量: {total_positive}")
    print(f"总负边数量: {total_negative}")
    print(f"总边数量: {total_edges}")
    print(f"\n文档统计:")
    print(f"总文档数量: {len(all_docs)}")
    print(f"有边的文档数量: {len(docs_with_edges)}")
    print(f"孤立文档数量: {len(isolated_docs)}")
    if isolated_docs:
        print("孤立文档列表:", sorted(isolated_docs))

    print(f"\n聚类统计:")
    print(f"聚类数量: {len(clusters)}")
    print("各聚类大小:")
    for i, (_, cluster) in enumerate(clusters.items(), 1):
        print(f"聚类 {i}: {len(cluster)} 个文档")


# 使用示例
count_document_edges('dataset/relations_gemma2_27b_0.009_0-1000.csv', 'edge_statistics.csv')
