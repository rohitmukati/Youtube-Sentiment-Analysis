import matplotlib
matplotlib.use('Agg')  # For non-GUI backends

from flask import Flask
from flask_cors import CORS
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Optional: download NLTK resources if not already done
import nltk

# ========== Preprocessing Function ==========
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return comment

# ========== Load Model and Vectorizer ==========
def load_model(model_path, vectorizer_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model/vectorizer: {e}")
        return None, None

# ========== Main Test ==========
if __name__ == "__main__":
    # File paths
    model_path = "./lgbm_model.pkl"
    vectorizer_path = "./tfidf_vectorizer.pkl"

    # Load model and vectorizer
    model, vectorizer = load_model(model_path, vectorizer_path)

    if model is None or vectorizer is None:
        print("Model or vectorizer loading failed.")
        exit(1)

    # Sample test comment
    sample_comments = ["Hi my name is Rohit"]

    # Preprocess
    preprocessed = [preprocess_comment(comment) for comment in sample_comments]
    print("Preprocessed:", preprocessed)

    # Vectorize
    transformed = vectorizer.transform(preprocessed)
    print("Transformed shape:", transformed)

    # Predict
    prediction = model.predict(transformed.toarray())
    print("Prediction:", prediction)
