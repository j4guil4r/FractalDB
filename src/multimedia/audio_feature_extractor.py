# src/multimedia/audio_feature_extractor.py
import os
import tempfile
from typing import Optional

import librosa
import numpy as np


class MFCCExtractor:
    """Extrae MFCC por frames desde un audio."""

    def __init__(self, sr: int = 22050, n_mfcc: int = 13, hop_length: int = 512):
        # sr: frecuencia de muestreo a la que se reescala el audio
        # n_mfcc: cantidad de coeficientes MFCC por frame
        # hop_length: salto entre frames (en muestras)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        print(f"[MFCCExtractor] sr={sr}, n_mfcc={n_mfcc}, hop_length={hop_length}")

    def extract_from_path(self, audio_path: str) -> Optional[np.ndarray]:
        """Carga un archivo de audio y devuelve matriz (frames, n_mfcc)."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"[MFCCExtractor] Error al cargar '{audio_path}': {e}")
            return None

        if y is None or y.size == 0:
            print(f"[MFCCExtractor] Audio vacío en '{audio_path}'")
            return None

        try:
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
            )
        except Exception as e:
            print(f"[MFCCExtractor] Error al calcular MFCC en '{audio_path}': {e}")
            return None

        if mfcc is None or mfcc.size == 0:
            print(f"[MFCCExtractor] MFCC vacío en '{audio_path}'")
            return None

        # Transponemos para obtener (frames, n_mfcc)
        return mfcc.T.astype(np.float32)

    def extract_from_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Guarda bytes de audio en un archivo temporal y reusa extract_from_path."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            return self.extract_from_path(tmp_path)
        except Exception as e:
            print(f"[MFCCExtractor] Error procesando bytes de audio: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
