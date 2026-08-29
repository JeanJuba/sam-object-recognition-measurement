"""Metrologia computacional (Fase 3).

A partir da máscara binária do SAM:

* ``cv2.minAreaRect``  -> retângulo delimitador mínimo -> comprimento e
  largura reais (mm), usando o fator de escala da calibração;
* contagem de pixels da máscara -> área real (mm²);
* comparação com tolerâncias do config -> classificação Aprovado/Reprovado.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Measurement:
    length_mm: float                 # maior dimensão do minAreaRect
    width_mm: float                  # menor dimensão do minAreaRect
    area_mm2: float                  # área da máscara
    rect: tuple                      # ((cx, cy), (w, h), ângulo) em pixels
    box_points: np.ndarray           # 4 vértices do retângulo (para desenho)


@dataclass
class Classification:
    class_name: str
    approved: bool
    reasons: list[str]               # motivos de reprovação (se houver)


def measure_mask(mask: np.ndarray, scale_mm_per_px: float) -> Measurement | None:
    """Extrai as métricas geométricas de uma máscara binária."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)          # ((cx, cy), (w, h), ângulo)
    (w_px, h_px) = rect[1]
    if w_px == 0 or h_px == 0:
        return None

    length_mm = max(w_px, h_px) * scale_mm_per_px
    width_mm = min(w_px, h_px) * scale_mm_per_px
    area_mm2 = float(cv2.countNonZero(mask)) * scale_mm_per_px**2

    return Measurement(
        length_mm=float(length_mm),
        width_mm=float(width_mm),
        area_mm2=float(area_mm2),
        rect=rect,
        box_points=cv2.boxPoints(rect).astype(np.int32),
    )


def median_measurement(history: list[Measurement], window: int = 15) -> Measurement:
    """Mediana móvel das últimas medições de um track — só para EXIBIÇÃO.

    Estabiliza os números do rótulo no vídeo (a borda da máscara treme
    ±1-2 px entre frames); a consolidação oficial do CSV continua sendo a
    mediana do track inteiro em ``report.summarize_tracks``.
    """
    recent = history[-window:]
    last = recent[-1]
    return Measurement(
        length_mm=float(np.median([m.length_mm for m in recent])),
        width_mm=float(np.median([m.width_mm for m in recent])),
        area_mm2=float(np.median([m.area_mm2 for m in recent])),
        rect=last.rect,                # geometria do frame atual,
        box_points=last.box_points,    # p/ posicionar o rótulo
    )


def classify(length_mm: float, width_mm: float, classes_cfg: list[dict]) -> Classification:
    """Atribui a peça à classe mais próxima e verifica as tolerâncias.

    A classe escolhida é a de menor distância entre as dimensões medidas e o
    centro nominal das faixas; em seguida verifica-se se comprimento e
    largura estão dentro de [min, max].
    """
    if not classes_cfg:
        return Classification(class_name="desconhecida", approved=False, reasons=["sem classes configuradas"])

    def nominal(cfg: dict) -> tuple[float, float]:
        lo_l, hi_l = cfg["length_mm"]
        lo_w, hi_w = cfg["width_mm"]
        return (lo_l + hi_l) / 2, (lo_w + hi_w) / 2

    best = min(
        classes_cfg,
        key=lambda c: np.hypot(length_mm - nominal(c)[0], width_mm - nominal(c)[1]),
    )

    reasons = []
    lo_l, hi_l = best["length_mm"]
    lo_w, hi_w = best["width_mm"]
    if not (lo_l <= length_mm <= hi_l):
        reasons.append(f"comprimento {length_mm:.1f}mm fora de [{lo_l}, {hi_l}]")
    if not (lo_w <= width_mm <= hi_w):
        reasons.append(f"largura {width_mm:.1f}mm fora de [{lo_w}, {hi_w}]")

    return Classification(class_name=best["name"], approved=not reasons, reasons=reasons)
