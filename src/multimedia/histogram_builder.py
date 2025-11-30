import numpy as np
from typing import Optional

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

from .feature_extractor import SIFTExtractor

from .audio_feature_extractor import  MFCCExtractor

from .codebook_builder import BaseCodebookBuilder



class BaseBoVWHistogramBuilder:
    """Generador de histogramas BoVW/BoAW genérico."""

    def __init__(
        self,
        k: int,
        data_dir: str,
        extractor,
        prefix: str,
    ):
        self.k = k
        self.data_dir = data_dir
        self.extractor = extractor

        self.kmeans: Optional[MiniBatchKMeans] = BaseCodebookBuilder.load_codebook(
            k, data_dir=data_dir, prefix=prefix
        )
        if self.kmeans is None:
            raise FileNotFoundError(
                f"No se pudo cargar el codebook para prefix={prefix}, K={k}."
            )

        self.tree = KDTree(self.kmeans.cluster_centers_, leaf_size=40)
        print(f"[Hist] prefix={prefix} K={k} listo.")

    def _create_histogram(self, descriptors: Optional[np.ndarray]) -> np.ndarray:
        if descriptors is None or descriptors.size == 0:
            return np.zeros(self.k, dtype=np.float32)

        _, visual_words = self.tree.query(descriptors, k=1)
        visual_words = visual_words.flatten()
        hist = np.bincount(visual_words, minlength=self.k)
        return hist.astype(np.float32)

    def create_histogram_from_path(self, path: str) -> Optional[np.ndarray]:
        des = self.extractor.extract_from_path(path)
        return self._create_histogram(des)

    def create_histogram_from_bytes(self, data: bytes) -> Optional[np.ndarray]:
        if not hasattr(self.extractor, "extract_from_bytes"):
            return None
        des = self.extractor.extract_from_bytes(data)
        return self._create_histogram(des)


class BoVWHistogramBuilder(BaseBoVWHistogramBuilder):
    """Histograma BoVW para imágenes (SIFT)."""

    def __init__(self, k: int, data_dir: str = "data"):
        super().__init__(
            k=k,
            data_dir=data_dir,
            extractor=SIFTExtractor(),
            prefix="mm",
        )

class AudioBoVWHistogramBuilder(BaseBoVWHistogramBuilder):
    """Histograma BoAW para audio (MFCC)."""

    def __init__(self, k: int, data_dir: str = "data"):
        super().__init__(
            k=k,
            data_dir=data_dir,
            extractor=MFCCExtractor(),
            prefix="mm_audio",
        )