"""BM25 over each photograph's own text (title, photographer, date): the search an archive has without a model."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from app.search.collection import Photograph

TERM_SATURATION = 1.2  # BM25 k1
LENGTH_NORMALISATION = 0.75  # BM25 b


def caption(photograph: Photograph) -> str:
    return f"{photograph.title} {photograph.artist} {photograph.date}"


class KeywordIndex:
    def __init__(self, photographs: list[Photograph]) -> None:
        self._vectorizer = CountVectorizer()  # words of two or more letters, case folded
        self._counts = self._vectorizer.fit_transform([caption(photograph) for photograph in photographs]).tocsc()
        document_lengths = np.asarray(self._counts.sum(axis=1)).ravel()
        self._length_factor = TERM_SATURATION * (
            1 - LENGTH_NORMALISATION + LENGTH_NORMALISATION * document_lengths / document_lengths.mean()
        )
        documents_with_term = np.asarray((self._counts > 0).sum(axis=0)).ravel()
        total = len(photographs)
        self._idf = np.log(1 + (total - documents_with_term + 0.5) / (documents_with_term + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """One BM25 score per photograph. Zero where no query word is in the caption."""
        terms = self._vectorizer.transform([query]).indices
        frequencies = self._counts[:, terms].toarray()
        saturated = frequencies * (TERM_SATURATION + 1) / (frequencies + self._length_factor[:, None])
        return np.asarray(saturated @ self._idf[terms]).ravel()
