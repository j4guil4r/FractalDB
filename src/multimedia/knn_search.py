import os
import pickle
import heapq
from typing import Iterable, Tuple, List, Any, Optional, Dict

import numpy as np

from .histogram_builder import BoVWHistogramBuilder, BaseBoVWHistogramBuilder, AudioBoVWHistogramBuilder


class BaseKNNSearch:
    """Búsqueda secuencial KNN genérica con TF-IDF y coseno."""

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

        self.db_path = os.path.join(data_dir, f"{prefix}_knn_seq_db_k{self.k}.pkl")

        self.idf_vector: Optional[np.ndarray] = None
        self.ids: List[Any] = []
        self.tfidf_vectors: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None

    def build_database(self, id_path_tuples: Iterable[Tuple[Any, str]]) -> None:
        print(f"[KNN] Construyendo BD secuencial prefix={self.prefix}...")

        raw_hists: Dict[Any, np.ndarray] = {}
        df = np.zeros(self.k, dtype=np.int32)
        doc_count = 0

        for obj_id, path in id_path_tuples:
            hist_tf = self.hist_builder.create_histogram_from_path(path)
            if hist_tf is not None and np.sum(hist_tf) > 0:
                raw_hists[obj_id] = hist_tf
                df[hist_tf > 0] += 1
                doc_count += 1
                if doc_count % 200 == 0:
                    print(f"[KNN] {doc_count} objetos procesados.")

        if doc_count == 0:
            print("[KNN] No se procesaron objetos.")
            return

        N = doc_count
        self.idf_vector = np.log((N + 1) / (df + 1)) + 1.0

        ids: List[Any] = []
        tfidf_list: List[np.ndarray] = []
        norms: List[float] = []

        for obj_id, hist_tf in raw_hists.items():
            tfidf_vec = hist_tf * self.idf_vector
            nrm = np.linalg.norm(tfidf_vec)
            if nrm <= 0:
                continue
            ids.append(obj_id)
            tfidf_list.append(tfidf_vec)
            norms.append(float(nrm))

        self.ids = ids
        self.tfidf_vectors = np.array(tfidf_list, dtype=np.float32)
        self.vector_norms = np.array(norms, dtype=np.float32)

        db_data = {
            "ids": self.ids,
            "tfidf_vectors": self.tfidf_vectors,
            "vector_norms": self.vector_norms,
            "idf_vector": self.idf_vector,
        }

        with open(self.db_path, "wb") as f:
            pickle.dump(db_data, f)

        print(f"[KNN] BD guardada en {self.db_path}")
        print(f"[KNN] Vectores: {self.tfidf_vectors.shape}")

    def load_database(self) -> None:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"[KNN] No existe BD: {self.db_path}")

        with open(self.db_path, "rb") as f:
            db = pickle.load(f)

        self.ids = db["ids"]
        self.tfidf_vectors = db["tfidf_vectors"]
        self.vector_norms = db["vector_norms"]
        self.idf_vector = db["idf_vector"]

        print(f"[KNN] BD cargada. {len(self.ids)} vectores.")

    def _search_hist(self, q_hist_tf: np.ndarray, top_k: int) -> List[Tuple[float, Any]]:
        if self.tfidf_vectors is None:
            self.load_database()

        if q_hist_tf is None or np.sum(q_hist_tf) == 0:
            return []

        q_tfidf = q_hist_tf * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf)
        if q_norm == 0:
            return []

        dot_products = np.dot(self.tfidf_vectors, q_tfidf)
        norm_products = self.vector_norms * q_norm
        sims = dot_products / norm_products

        heap: List[Tuple[float, Any]] = []
        for i, score in enumerate(sims):
            obj_id = self.ids[i]
            if len(heap) < top_k:
                heapq.heappush(heap, (float(score), obj_id))
            else:
                heapq.heappushpop(heap, (float(score), obj_id))

        return sorted(heap, reverse=True)

    def search_by_path(self, path: str, top_k: int = 10) -> List[Tuple[float, Any]]:
        hist_tf = self.hist_builder.create_histogram_from_path(path)
        return self._search_hist(hist_tf, top_k)

    def search_by_bytes(self, data: bytes, top_k: int = 10) -> List[Tuple[float, Any]]:
        hist_tf = self.hist_builder.create_histogram_from_bytes(data)
        return self._search_hist(hist_tf, top_k)


class KNNSearch(BaseKNNSearch):
    """KNN secuencial para imágenes."""

    def __init__(self, k_clusters: int, data_dir: str = "data"):
        hist_builder = BoVWHistogramBuilder(k_clusters, data_dir)
        super().__init__(
            k_clusters=k_clusters,
            data_dir=data_dir,
            hist_builder=hist_builder,
            prefix="mm",
        )



class AudioKNNSearch(BaseKNNSearch):
    """KNN secuencial para audio."""

    def __init__(self, k_clusters: int, data_dir: str = "data"):
        hist_builder = AudioBoVWHistogramBuilder(k_clusters, data_dir)
        super().__init__(
            k_clusters=k_clusters,
            data_dir=data_dir,
            hist_builder=hist_builder,
            prefix="mm_audio",
        )
