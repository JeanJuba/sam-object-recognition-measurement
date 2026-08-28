"""Desenho das anotações no vídeo de saída."""

import cv2
import numpy as np

from .metrology import Classification, Measurement

GREEN = (0, 200, 0)
RED = (0, 0, 255)
BLUE = (255, 160, 0)
WHITE = (255, 255, 255)

FONT_SCALE = 0.6
LINE_SPACING = 20


def _put_label(frame: np.ndarray, lines: list[str], x: int, y: int, color: tuple) -> None:
    """Escreve linhas de texto no frame."""
    for i, line in enumerate(lines):
        pos = (x, y + i * LINE_SPACING)
        cv2.putText(frame, line, pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 1, cv2.LINE_AA)


def draw_object(
    frame: np.ndarray,
    mask: np.ndarray,
    measurement: Measurement,
    classification: Classification,
    track_id: int,
) -> None:
    """Desenha máscara, retângulo mínimo e rótulo de uma peça (in-place)."""
    color = GREEN if classification.approved else RED

    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

    cv2.drawContours(frame, [measurement.box_points], 0, color, 2)

    status = "APROVADO" if classification.approved else "REPROVADO"
    lines = [
        f"#{track_id} {classification.class_name} [{status}]",
        f"C: {measurement.length_mm:.1f}mm  L: {measurement.width_mm:.1f}mm",
        f"Area: {measurement.area_mm2:.0f}mm2",
    ]
    x, y = measurement.box_points.min(axis=0)
    y = max(int(y) - 8, 15)
    _put_label(frame, lines, int(x), y, WHITE)


def draw_reference(
    frame: np.ndarray,
    mask: np.ndarray,
    measurement: Measurement,
    track_id: int,
) -> None:
    """Desenha a referência móvel (1º objeto da esteira) em azul, sem classificar.

    Mostra a medida ao vivo — deve bater com a dimensão conhecida do config,
    servindo de verificação visual da escala.
    """
    overlay = frame.copy()
    overlay[mask > 0] = BLUE
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

    cv2.drawContours(frame, [measurement.box_points], 0, BLUE, 2)

    lines = [
        f"#{track_id} REF",
        f"C: {measurement.length_mm:.1f}mm",
    ]
    x, y = measurement.box_points.min(axis=0)
    y = max(int(y) - 8, 15)
    _put_label(frame, lines, int(x), y, BLUE)


def draw_calibration(frame: np.ndarray, center: tuple | None, radius_px: float | None, scale: float) -> None:
    """Marca o objeto de referência e mostra o fator de escala."""
    if center is not None and radius_px is not None:
        cv2.circle(frame, center, int(radius_px), BLUE, 2)
        cv2.putText(frame, "REF", (center[0] - 15, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2, cv2.LINE_AA)
    _put_label(frame, [f"Escala: {scale:.4f} mm/px"], 10, 25, WHITE)
