"""Filtros digitais de pré-processamento (Fase 1).

Suaviza texturas irrelevantes e reduz ruído do sensor antes da
segmentação, tornando os prompts do SAM mais estáveis.
"""

import cv2
import numpy as np


def apply_filter(frame: np.ndarray, filter_type: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
    """Aplica um filtro espacial ao frame.

    Args:
        frame: imagem BGR.
        filter_type: "gaussian", "mean" ou "none".
        kernel_size: tamanho do kernel (ímpar).
    """
    if filter_type == "none":
        return frame
    if kernel_size % 2 == 0:
        kernel_size += 1
    if filter_type == "gaussian":
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    if filter_type == "mean":
        return cv2.blur(frame, (kernel_size, kernel_size))
    raise ValueError(f"Filtro desconhecido: {filter_type!r}")
