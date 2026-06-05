"""Quick exploration of the MBTI dataset."""
import pandas as pd

df = pd.read_csv("data/MBTI_500.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print()

# Show first 5 rows
print("=== First 5 rows ===")
for i in range(5):
    print(f"--- Row {i} ---")
    print(f"type: {df.iloc[i]['type']}")
    posts_text = str(df.iloc[i]["posts"])
    print(f"posts length: {len(posts_text)} chars")
    print(f"posts (first 300 chars): {posts_text[:300]}")
    print()

print("\n=== Type Distribution ===")
print(df["type"].value_counts().to_string())

print("\n=== Null Counts ===")
print(df.isnull().sum().to_string())

print("\n=== Text Length Stats ===")
print(df["posts"].str.len().describe().to_string())
