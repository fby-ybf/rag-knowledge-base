"""向量化：默认 TF-IDF（离线可复现），接口与语义向量实现保持一致。"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """把文本编码为 L2 归一化的 TF-IDF 向量。

    生产环境可替换为 sentence-transformers / bge 等语义向量模型，
    只要保证 `transform` 返回二维 numpy 数组即可被 store.py 直接使用。
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        return self._normalize(matrix.toarray().astype("float32"))

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return self._normalize(matrix.toarray().astype("float32"))

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
