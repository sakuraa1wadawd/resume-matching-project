from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - fallback keeps demos runnable before dependencies are installed
    TfidfVectorizer = None

    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
        denom = np.maximum(left_norm @ right_norm, 1e-12)
        return (left @ right.T) / denom


def _clip_score(matrix: np.ndarray) -> np.ndarray:
    return np.clip(matrix * 100, 0, 100).round(2)


def compute_tfidf_scores(resume_texts: Sequence[str], job_texts: Sequence[str]) -> np.ndarray:
    if TfidfVectorizer is None:
        return _manual_tfidf_scores(resume_texts, job_texts)
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    all_texts = list(resume_texts) + list(job_texts)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    resume_matrix = tfidf_matrix[: len(resume_texts)]
    job_matrix = tfidf_matrix[len(resume_texts) :]
    return _clip_score(cosine_similarity(resume_matrix, job_matrix))


def _manual_tfidf_scores(resume_texts: Sequence[str], job_texts: Sequence[str]) -> np.ndarray:
    documents = [text.split() for text in list(resume_texts) + list(job_texts)]
    vocabulary = sorted({token for doc in documents for token in doc})
    if not vocabulary:
        return np.zeros((len(resume_texts), len(job_texts)))

    doc_count = len(documents)
    df = Counter(token for doc in documents for token in set(doc))
    idf = {token: math.log((1 + doc_count) / (1 + df[token])) + 1 for token in vocabulary}
    index = {token: idx for idx, token in enumerate(vocabulary)}
    matrix = np.zeros((doc_count, len(vocabulary)))

    for row_idx, doc in enumerate(documents):
        counts = Counter(doc)
        total = max(sum(counts.values()), 1)
        for token, count in counts.items():
            matrix[row_idx, index[token]] = (count / total) * idf[token]

    resume_matrix = matrix[: len(resume_texts)]
    job_matrix = matrix[len(resume_texts) :]
    return _clip_score(cosine_similarity(resume_matrix, job_matrix))


def _average_vectors(tokens: Sequence[str], model, vector_size: int) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def _deterministic_token_vector(token: str, vector_size: int) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((vector_size // len(digest)) + 1))[:vector_size], dtype=np.uint8)
    return (values.astype(float) - 127.5) / 127.5


def _fallback_word_vectors(token_lists: Sequence[Sequence[str]], vector_size: int) -> list[np.ndarray]:
    """Small offline fallback used only when gensim is unavailable."""
    vectors = []
    for tokens in token_lists:
        token_vectors = [_deterministic_token_vector(token, vector_size) for token in tokens]
        if token_vectors:
            vectors.append(np.mean(token_vectors, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return vectors


def compute_word2vec_scores(
    resume_tokens: Sequence[Sequence[str]],
    job_tokens: Sequence[Sequence[str]],
    vector_size: int = 80,
) -> tuple[np.ndarray, str]:
    token_lists = list(resume_tokens) + list(job_tokens)
    try:
        from gensim.models import Word2Vec

        model = Word2Vec(
            sentences=token_lists,
            vector_size=vector_size,
            window=4,
            min_count=1,
            workers=1,
            sg=1,
            seed=42,
            epochs=120,
        )
        vectors = [_average_vectors(tokens, model, vector_size) for tokens in token_lists]
        method = "gensim Word2Vec"
    except Exception:
        vectors = _fallback_word_vectors(token_lists, vector_size)
        method = "deterministic fallback vectors"

    resume_vectors = np.vstack(vectors[: len(resume_tokens)])
    job_vectors = np.vstack(vectors[len(resume_tokens) :])
    return _clip_score(cosine_similarity(resume_vectors, job_vectors)), method


def compute_semantic_scores(tfidf_scores: np.ndarray, word2vec_scores: np.ndarray) -> np.ndarray:
    return np.clip(tfidf_scores * 0.55 + word2vec_scores * 0.45, 0, 100).round(2)
