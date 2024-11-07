# import pandas as pd

# # Load the claims file (claims_20241106_184644.csv)
# claims_df = pd.read_csv('./dataset/claims_20241106_184644.csv')

# # Ensure that 'claim_id' is a string and zero-padded to 8 digits
# claims_df['claim_id'] = claims_df['claim_id'].apply(lambda x: str(x).zfill(8))

# # Create a mapping from the row index to the claim ID using integers as keys
# id_mapping = {i: claims_df.iloc[i]['claim_id'] for i in range(len(claims_df))}

# # Load the claim similarity results (claim_similarity_results.csv)
# similarity_df = pd.read_csv('./dataset/claim_similarity_results.csv')

# # Replace id1 and id2 with their corresponding claim IDs from the mapping
# similarity_df['id1'] = similarity_df['id1'].map(id_mapping)
# similarity_df['id2'] = similarity_df['id2'].map(id_mapping)

# # Print the modified similarity dataframe
# print(similarity_df)

# # Optionally, save the updated results to a new CSV
# similarity_df = sorted(similarity_df, key=lambda x: x[2], reverse=True)
# similarity_df.to_csv('./dataset/updated_claim_similarity_results.csv', index=False)

import pandas as pd

# Load the CSV file
df = pd.read_csv('./dataset/claim_similarity_results.csv')

df['id1'] = df['id1'].astype(str).str.zfill(8)
df['id2'] = df['id2'].astype(str).str.zfill(8)

# Sort the DataFrame by the third column (index 2)
df_sorted = df.sort_values(by=df.columns[2], ascending=False)


# Keep only the first 1000 rows
df_first_1000 = df_sorted.head(1000)

# Save the result to a new CSV file
df_first_1000.to_csv('./dataset/filtered_claim_similarity_results.csv', index=False)
df_sorted.to_csv('./dataset/sorted_claim_similarity_results.csv', index=False)

