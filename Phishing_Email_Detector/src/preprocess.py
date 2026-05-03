import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """clean the text by removing stopwords and punctuation and converting to lowercase"""
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    return ' '.join([ word for word in text.split() if word not in stop_words ])

def preprocess_df(df):
    """preprocess the dataframe by cleaning the text and encoding the labels"""
    df['clean_text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Phishing Email' else 0)
    return df