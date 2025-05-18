import pandas as pd

# Load your new dataset
df = pd.read_csv("reviews.csv")

# Show original columns
print("📋 Original columns:", df.columns.tolist(), "\n")

# Merge Summary and Review into one column
df['Review'] = df['Summary'].astype(str) + " " + df['Review'].astype(str)

# Standardize sentiment labels
df['Sentiment'] = df['Sentiment'].str.capitalize()

# Keep only the required columns
df = df[['Review', 'Sentiment']]

# Show cleaned data preview
print("✅ Cleaned dataset preview:\n")
print(df.head(10))

# Show sentiment distribution
print("\n📊 Sentiment distribution:")
print(df['Sentiment'].value_counts())

# Save cleaned dataset
df.to_csv("cleaned_reviews.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_reviews.csv'")
