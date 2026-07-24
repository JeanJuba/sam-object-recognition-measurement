"""Geração de relatórios e análise de erro (Fase 3 / Relatório final).

* Consolida as medições por peça (mediana ao longo dos frames) em CSV;
* Compara com medições manuais (paquímetro) e calcula MAE e MSE.
"""

import numpy as np
import pandas as pd

from .metrology import classify
from .tracking import Track


def summarize_tracks(tracks: list[Track], classes_cfg: list[dict]) -> pd.DataFrame:
    """Consolida cada track em uma linha: mediana das medições por peça.

    A mediana ao longo dos frames reduz o efeito de máscaras ruins em
    frames isolados.
    """
    rows = []
    for track in sorted(tracks, key=lambda t: t.track_id):
        if not track.measurements:
            continue
        lengths = [m.length_mm for m in track.measurements]
        widths = [m.width_mm for m in track.measurements]
        areas = [m.area_mm2 for m in track.measurements]

        length = float(np.median(lengths))
        width = float(np.median(widths))
        cls = classify(length, width, classes_cfg)

        rows.append({
            "track_id": track.track_id,
            "classe": cls.class_name,
            "comprimento_mm": round(length, 2),
            "largura_mm": round(width, 2),
            "area_mm2": round(float(np.median(areas)), 1),
            "desvio_comprimento_mm": round(float(np.std(lengths)), 2),
            "desvio_largura_mm": round(float(np.std(widths)), 2),
            "n_frames_medidos": len(track.measurements),
            "status": "APROVADO" if cls.approved else "REPROVADO",
            "motivos_reprovacao": "; ".join(cls.reasons),
        })
    return pd.DataFrame(rows)


def error_analysis(measured_csv: str, ground_truth_csv: str) -> tuple[pd.DataFrame, dict]:
    """Compara medições do software com as medições manuais (paquímetro).

    O CSV de ground truth deve ter as colunas:
        track_id, comprimento_real_mm, largura_real_mm

    Returns:
        (tabela comparativa, {métrica: valor}) com MAE e MSE por dimensão.
    """
    measured = pd.read_csv(measured_csv)
    truth = pd.read_csv(ground_truth_csv)
    df = measured.merge(truth, on="track_id", how="inner")
    if df.empty:
        raise ValueError("Nenhum track_id em comum entre medições e ground truth.")

    df["erro_comprimento_mm"] = df["comprimento_mm"] - df["comprimento_real_mm"]
    df["erro_largura_mm"] = df["largura_mm"] - df["largura_real_mm"]

    metrics = {}
    for dim in ("comprimento", "largura"):
        err = df[f"erro_{dim}_mm"]
        metrics[f"MAE_{dim}_mm"] = float(err.abs().mean())
        metrics[f"MSE_{dim}_mm2"] = float((err**2).mean())
        metrics[f"RMSE_{dim}_mm"] = float(np.sqrt((err**2).mean()))

    cols = [
        "track_id", "classe",
        "comprimento_real_mm", "comprimento_mm", "erro_comprimento_mm",
        "largura_real_mm", "largura_mm", "erro_largura_mm",
        "status",
    ]
    return df[[c for c in cols if c in df.columns]], metrics
