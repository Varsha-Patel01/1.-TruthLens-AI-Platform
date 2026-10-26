
import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("\nFAKE NEWS SUBJECTS:")
print(fake["subject"].value_counts())

print("\nREAL NEWS SUBJECTS:")
print(true["subject"].value_counts())

print("\nFAKE NEWS EXAMPLES:")
print(fake[["title", "subject"]].head(5).to_string(index=False))

print("\nREAL NEWS EXAMPLES:")
print(true[["title", "subject"]].head(5).to_string(index=False))
