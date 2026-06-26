import pandas as pd
import re

# 1. Load data y
df = pd.read_csv('reviews_shopee.csv')

def clean_text(text):
    # Change to string and lower case
    text = str(text).lower()
    # Remove URL/Link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mention (@) and hashtag (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

# 2. Run the cleaning function
df['content_cleaned'] = df['content'].apply(clean_text)

# 3. Create Simple Sentiment Labels (Pseudo-labeling)
# Score 4-5 = Positive (1), Score 1-2 = Negative (0), Score 3 = Neutral (ignore this for now to focus)
def create_label(score):
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return None # Netral

df['label'] = df['score'].apply(create_label)

# Remove the neutral so that the model learns contrasting differences.
df = df.dropna(subset=['label'])

# 4. Save the clean data to a new CSV file
df[['content_cleaned', 'label']].to_csv('shopee_reviews_cleaned.csv', index=False)

print("Data sudah bersih dan siap diolah!")
print(df[['content_cleaned', 'label']].head())