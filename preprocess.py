import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt")
nltk.download('punkt_tab')

# Load pretrained FastText vectors
FASTTEXT_PATH = r"C:\Users\Lenovo\Desktop\XAI\SentX\embeddings\Fasttext\wiki-news-300d-1M-subword.vec"

# === Load FastText model ===
fasttext_model = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)

# === Text cleaning ===
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

# === Tokenize and vectorize review ===
def review_to_token_vectors(text, max_len=30):
    tokens = word_tokenize(clean_text(text))
    vectors = []

    for token in tokens:
        if token in fasttext_model:
            vectors.append(fasttext_model[token])
        else:
            vectors.append(np.zeros(fasttext_model.vector_size))  # unknown token

    # Pad or truncate to max_len
    vectors = vectors[:max_len] + [np.zeros(fasttext_model.vector_size)] * max(0, max_len - len(vectors))
    return np.array(vectors, dtype=np.float32)

# === Batch version
def encode_reviews_bilstm(reviews, max_len=30):
    encoded_reviews = []

    for r in reviews:
        try:
            vec = review_to_token_vectors(r, max_len)
            if vec.shape != (max_len, fasttext_model.vector_size):
                raise ValueError("Shape mismatch")
            encoded_reviews.append(vec)
        except Exception as e:
            # If tokenization fails, append zero-matrix
            encoded_reviews.append(np.zeros((max_len, fasttext_model.vector_size)))

    return np.array(encoded_reviews, dtype=np.float32)

