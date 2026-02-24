import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

#  VERY IMPORTANT: Keep negation words
negation_words = {"no", "not", "nor", "never", "cannot"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    cleaned_words = []
    for w in words:
        if w not in stop_words:
            cleaned_words.append(lemmatizer.lemmatize(w))

    return ' '.join(cleaned_words)


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df = df[['review', 'rating']]
    df.dropna(inplace=True)

    df['clean_review'] = df['review'].apply(clean_text)

    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 7 else 0)

    return df['clean_review'], df['sentiment']