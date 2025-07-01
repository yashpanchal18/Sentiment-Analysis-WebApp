import shap
import joblib
import numpy as np
import re
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from preprocess import clean_text, review_to_token_vectors
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from shap import Explainer, maskers
import warnings

warnings.filterwarnings("ignore")

# === Load model and components ===
model = load_model(r"C:\Users\Lenovo\Desktop\XAI\SentX\models\lstm_sentiment_model.h5")
label_encoder = joblib.load(r"C:\Users\Lenovo\Desktop\XAI\SentX\models\lstm_label_encoder.pkl")
fasttext_model = KeyedVectors.load_word2vec_format(
    r"C:\Users\Lenovo\Desktop\XAI\SentX\embeddings\Fasttext\wiki-news-300d-1M-subword.vec", binary=False
)

# === Token Normalization Helper ===
def normalize_token(token):
    return re.sub(r'\W+', '', token.strip().lower())

# === SHAP Explanation ===
def shap_explain(text, top_k=10):
    def predict_from_tokens(texts):
        encoded = np.array([
            review_to_token_vectors(clean_text(t), max_len=30) for t in texts
        ], dtype=np.float32)
        return model.predict(encoded, verbose=0)

    masker = maskers.Text()
    explainer = Explainer(predict_from_tokens, masker)
    shap_values = explainer([text])

    pred_probs = predict_from_tokens([text])
    pred_class = np.argmax(pred_probs[0])
    scores = shap_values[0].values[:, pred_class]
    tokens = shap_values[0].data

    top_tokens = sorted(
        zip(tokens, scores),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    return {normalize_token(tok): round(score, 4) for tok, score in top_tokens}

# === LIME Explanation ===
def lime_explain(text, top_k=10):
    def predict_proba(texts):
        encoded = np.array([
            review_to_token_vectors(clean_text(t), max_len=30) for t in texts
        ], dtype=np.float32)
        return model.predict(encoded, verbose=0)

    explainer = LimeTextExplainer(class_names=label_encoder.classes_.tolist())
    exp = explainer.explain_instance(text, predict_proba, num_features=top_k, num_samples=1000)
    return {normalize_token(tok): round(score, 4) for tok, score in exp.as_list()}

# Add this function to the existing code, right before compare_explanations()

def combine_explanations(lime_scores, shap_scores, strategy="auto", model_confidence=None, disable_filter=False):
    print(f"\n🧠 Combining explanations with strategy: {strategy} | Confidence: {round(model_confidence, 4)}")

    hybrid_scores = {}
    conflicts = []
    all_tokens = set(lime_scores.keys()).union(set(shap_scores.keys()))

    for token in all_tokens:
        lime_val = lime_scores.get(token, 0.0)
        shap_val = shap_scores.get(token, 0.0)

        if strategy == "auto":
            if np.sign(lime_val) != np.sign(shap_val) and abs(lime_val) > 0.01 and abs(shap_val) > 0.01:
                conflicts.append(f"{token} (resolved via average)")
                score = (lime_val + shap_val) / 2
            else:
                score = lime_val + shap_val
        else:
            score = lime_val + shap_val  # Fallback
        hybrid_scores[token] = round(score, 4)

    if conflicts:
        print(f"\n⚠️ Conflicts Detected: {conflicts}")

    if not disable_filter:
        filtered = [(tok, score) for tok, score in hybrid_scores.items() if abs(score) < 0.01]
        for tok, _ in filtered:
            del hybrid_scores[tok]
        if filtered:
            print(f"\n🧹 Filtered out (low-impact < 0.01): {filtered}")

    return hybrid_scores




# === Hybrid Overlap + Normalized Analysis ===
def compare_explanations(text, top_k=10):
    print(f"\n📄 Input Review: {text}")
    lime_top = lime_explain(text, top_k)
    shap_top = shap_explain(text, top_k)


    # Combine scores using hybrid logic
    # Get model confidence
    input_vec = np.array([review_to_token_vectors(clean_text(text), max_len=30)], dtype=np.float32)
    prediction_probs = model.predict(input_vec, verbose=0)[0]
    confidence = float(np.max(prediction_probs))

    hybrid_top = combine_explanations(lime_top, shap_top, model_confidence=confidence, strategy="auto")
    print(f"\n🤖 Model Confidence: {round(confidence, 4)}")

    print(f"\n🧠 Hybrid Token Scores (Top {top_k} by abs importance):")
    top_combined = sorted(hybrid_top.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    for tok, score in top_combined:
        print(f"  {tok:15} | Hybrid Score: {score:6.4f}")


    lime_tokens = set(lime_top.keys())
    shap_tokens = set(shap_top.keys())

    intersection = lime_tokens & shap_tokens
    union = lime_tokens | shap_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    print(f"\n✅ LIME Top-{top_k} Tokens: {list(lime_top.items())}")
    print(f"✅ SHAP Top-{top_k} Tokens: {list(shap_top.items())}")
    print(f"🔁 Overlapping Tokens (normalized): {intersection}")
    print(f"📊 Jaccard Similarity: {round(jaccard, 4)}")

    print("\n🔍 Token-wise Comparison (normalized):")
    for token in union:
        lime_score = lime_top.get(token, 0.0)
        shap_score = shap_top.get(token, 0.0)
        print(f"  {token:15} | LIME: {lime_score:6.4f} | SHAP: {shap_score:6.4f}")

# Add at the bottom of hybrid.py
__all__ = ['shap_explain', 'lime_explain', 'combine_explanations', 'model', 'label_encoder']


# === Example usage ===
if __name__ == "__main__":
    test_review = "Battery life is decent but the build quality is awful."
    compare_explanations(test_review, top_k=10)
