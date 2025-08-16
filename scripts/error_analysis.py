# scripts/error_analysis.py

import pandas as pd

df = pd.read_csv("results/eval_results.csv")

print("🔍 Generated responses:\n")
print(df[["question", "expected", "predicted"]])

# Check for exact match failures
failures = df[df["exact_match"] == 0]

print(f"\n❌ Number of failed predictions: {len(failures)}")
if len(failures) > 0:
    print("\n🧠 Mismatches:\n")
    print(failures[["question", "expected", "predicted"]])
else:
    print("✅ All predictions matched the expected answers!")
