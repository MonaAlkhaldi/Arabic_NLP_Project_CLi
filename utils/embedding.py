from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from model2vec import StaticModel


# ----------------------------
# TF-IDF
# ----------------------------
def build_tfidf(
    texts: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
):
    """
    Fit TF-IDF and transform texts into a sparse matrix.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False,             # Arabic: no lower/upper
        token_pattern=r"(?u)\b\w+\b" # supports Arabic tokens
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def save_tfidf(X, vectorizer, out_dir: str | Path, prefix: str = "tfidf"):
    # ضع كل ملفات TF-IDF داخل مجلد embeddings
    out_dir = Path(out_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / f"{prefix}_vectors.npz"
    vec_path = out_dir / f"{prefix}_vectorizer.pkl"

    sp.save_npz(X_path, X)
    joblib.dump(vectorizer, vec_path)
    return X_path, vec_path



# ----------------------------
# Model2Vec (ARBERTv2)
# ----------------------------
def build_model2vec(
    texts: list[str],
    model_name: str = "JadwalAlmaa/model2vec-ARBERTv2",
):
    """
    Generate sentence embeddings using Model2Vec.
    """
    model = StaticModel.from_pretrained(model_name)
    emb = model.encode(texts)  # numpy array (n_samples, dim)
    return emb, model


def save_model2vec(embeddings: np.ndarray, model: StaticModel, out_dir: str | Path, prefix: str = "model2vec"):
    # ضع كل ملفات Model2Vec داخل مجلد embeddings
    out_dir = Path(out_dir) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"{prefix}_embeddings.npy"
    model_path = out_dir / f"{prefix}_model.pkl"

    np.save(emb_path, embeddings)
    joblib.dump(model, model_path)
    return emb_path, model_path
