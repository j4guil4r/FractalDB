import os
import pickle
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from .histogram_builder import BoVWHistogramBuilder, BaseBoVWHistogramBuilder,AudioBoVWHistogramBuilder


class BaseMMInvertedIndexQuery:
    """Consulta KNN indexada genérica sobre un índice invertido multimedia."""

    def __init__(
        self,
        k_clusters: int,
        data_dir: str,
        hist_builder: BaseBoVWHistogramBuilder,
        prefix: str,
    ):
        self.k = k_clusters
        self.data_dir = data_dir
        self.hist_builder = hist_builder
        self.prefix = prefix

        self.index_path = os.path.join(
            data_dir, f"{prefix}_inverted_index_k{self.k}.dat"
        )
        self.meta_path = os.path.join(
            data_dir, f"{prefix}_inverted_index_k{self.k}.meta"
        )

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"Índice {prefix} (k={self.k}) no encontrado. Construya el índice primero."
            )

        self._load_metadata()

        try:
            self.index_file = open(self.index_path, "rb")
        except IOError as e:
            print(f"[MMQuery] Error al abrir índice: {e}")
            raise

        print(f"[MMQuery] prefix={self.prefix} K={self.k}, docs={self.total_docs}")

    def _load_metadata(self) -> None:
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)

        self.k = meta["k"]
        self.total_docs: int = meta["total_docs"]
        self.doc_metadata: Dict[Any, Tuple[float, float]] = meta["doc_metadata"]
        self.lexicon: Dict[int, Tuple[int, int]] = meta["lexicon"]
        self.idf_vector: np.ndarray = meta["idf_vector"]

    def _get_postings(self, term_id: int) -> List[Tuple[Any, float]]:
        lookup = self.lexicon.get(term_id)
        if not lookup:
            return []
        offset, length = lookup
        try:
            self.index_file.seek(offset)
            data = self.index_file.read(length)
            return pickle.loads(data)
        except Exception as e:
            print(f"[MMQuery] Error leyendo postings para {term_id}: {e}")
            return []

    def close(self) -> None:
        if hasattr(self, "index_file") and self.index_file:
            self.index_file.close()
            print("[MMQuery] Archivo cerrado.")

    def __del__(self) -> None:
        self.close()

    def _calculate_similarity(
        self,
        q_hist_tf: np.ndarray,
        top_k: int,
    ) -> List[Tuple[float, Any]]:
        if q_hist_tf is None or np.sum(q_hist_tf) == 0:
            return []

        q_tfidf = q_hist_tf * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf)
        if q_norm == 0:
            return []

        scores = defaultdict(float)
        query_ids = np.nonzero(q_hist_tf)[0]

        for term_id in query_ids:
            q_weight = q_tfidf[term_id]
            postings = self._get_postings(int(term_id))
            for doc_id, d_weight in postings:
                scores[doc_id] += q_weight * d_weight

        heap: List[Tuple[float, Any]] = []
        for doc_id, dot in scores.items():
            d_norm = self.doc_metadata[doc_id][1]
            if d_norm <= 0:
                continue
            sim = dot / (q_norm * d_norm)
            if len(heap) < top_k:
                heapq.heappush(heap, (sim, doc_id))
            else:
                heapq.heappushpop(heap, (sim, doc_id))

        return sorted(heap, reverse=True)

    def query_by_path(self, path: str, top_k: int = 10) -> List[Tuple[float, Any]]:
        hist = self.hist_builder.create_histogram_from_path(path)
        return self._calculate_similarity(hist, top_k)

    def query_by_bytes(self, data: bytes, top_k: int = 10) -> List[Tuple[float, Any]]:
        hist = self.hist_builder.create_histogram_from_bytes(data)
        return self._calculate_similarity(hist, top_k)


class MMInvertedIndexQuery(BaseMMInvertedIndexQuery):
    """Consulta KNN indexada para imágenes."""

    def __init__(self, k_clusters: int, data_dir: str = "data"):
        hist_builder = BoVWHistogramBuilder(k_clusters, data_dir)
        super().__init__(
            k_clusters=k_clusters,
            data_dir=data_dir,
            hist_builder=hist_builder,
            prefix="mm",
        )


class AudioMMInvertedIndexQuery(BaseMMInvertedIndexQuery):
    """Consulta KNN indexada para audio."""

    def __init__(self, k_clusters: int, data_dir: str = "data"):
        hist_builder = AudioBoVWHistogramBuilder(k_clusters, data_dir)
        super().__init__(
            k_clusters=k_clusters,
            data_dir=data_dir,
            hist_builder=hist_builder,
            prefix="mm_audio",
        )
