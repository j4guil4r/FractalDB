import os
import pickle
from typing import Iterable, Optional, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .feature_extractor import SIFTExtractor

from .audio_feature_extractor import MFCCExtractor


class BaseCodebookBuilder:
    """Codebook genérico BoVW/BoAW. Reutilizado por imagen y audio."""

    def __init__(
        self,
        k: int,
        data_dir: str,
        extractor,
        feature_dim: int,
        prefix: str,
        batch_size: int,
    ):
        self.k = k
        self.data_dir = data_dir
        self.extractor = extractor
        self.feature_dim = feature_dim
        self.prefix = prefix
        self.codebook_path = os.path.join(data_dir, f"{prefix}_codebook_k{k}.pkl")

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            verbose=True,
            batch_size=batch_size,
            n_init=3,
            max_iter=100,
            random_state=42,
        )

    def _get_descriptor_batches(
        self,
        paths: Iterable[str],
        batch_size: int,
    ) -> Iterable[np.ndarray]:
        desc_batch: List[np.ndarray] = []
        count = 0

        for path in paths:
            des = self.extractor.extract_from_path(path)
            if des is not None and des.size > 0:
                desc_batch.append(des)
                count += des.shape[0]

            if count >= batch_size and desc_batch:
                yield np.vstack(desc_batch)
                desc_batch = []
                count = 0

        if desc_batch:
            yield np.vstack(desc_batch)

    def build_from_paths(
        self,
        paths: Iterable[str],
        sample_limit: int = 500_000,
    ) -> None:
        print(f"[Codebook] prefix={self.prefix} K={self.k} sample_limit={sample_limit}")
        total = 0
        for i, batch in enumerate(self._get_descriptor_batches(paths, batch_size=4096)):
            if total >= sample_limit:
                print("[Codebook] Límite de muestras alcanzado.")
                break
            if batch is None or len(batch) <= self.k:
                continue

            idx = np.random.permutation(len(batch))[:min(2048, len(batch))]
            sample = batch[idx]
            self.kmeans.partial_fit(sample)
            total += len(sample)
            print(f"[Codebook] Lote {i + 1}, total descriptores={total}")

        if total == 0:
            print("[Codebook] No se extrajeron descriptores, ajustando codebook vacío.")
            self.kmeans.fit(np.zeros((self.k, self.feature_dim), dtype=np.float32))

        self.save_codebook()

    def save_codebook(self) -> None:
        print(f"[Codebook] Guardando en {self.codebook_path}")
        with open(self.codebook_path, "wb") as f:
            pickle.dump(self.kmeans, f)

    @staticmethod
    def load_codebook(
        k: int,
        data_dir: str = "data",
        prefix: str = "mm",
    ) -> Optional[MiniBatchKMeans]:
        path = os.path.join(data_dir, f"{prefix}_codebook_k{k}.pkl")
        if not os.path.exists(path):
            print(f"[Codebook] No se encontró {path}")
            return None
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"[Codebook] Cargado K={k} desde {path}")
            return model
        except Exception as e:
            print(f"[Codebook] Error al cargar codebook: {e}")
            return None


class CodebookBuilder(BaseCodebookBuilder):
    """Codebook para imágenes (SIFT, BoVW)."""

    def __init__(self, k: int, data_dir: str = "data"):
        super().__init__(
            k=k,
            data_dir=data_dir,
            extractor=SIFTExtractor(),
            feature_dim=128,
            prefix="mm",
            batch_size=2048,
        )

class AudioCodebookBuilder(BaseCodebookBuilder):
    """Codebook acústico (BoAW) usando MFCC."""

    def __init__(self, k: int, data_dir: str = "data"):
        super().__init__(
            k=k,
            data_dir=data_dir,
            extractor=MFCCExtractor(),
            feature_dim=13,
            prefix="mm_audio",
            batch_size=4096,
        )