"""Gera um vídeo sintético para validar o pipeline antes do vídeo real.

Simula uma esteira vista de cima com:
* um "pendrive" de referência fixo (retângulo de 55 mm) no canto
  inferior esquerdo, como no vídeo real;
* três classes de peças (retângulo grande, quadrado médio e quadrado
  pequeno) atravessando a cena, algumas fora de tolerância de propósito.

Como as dimensões reais são definidas aqui em mm, o vídeo também gera o
ground truth (data/ground_truth_sintetico.csv) para testar o MAE/MSE.
"""

import csv
import os

import cv2
import numpy as np

W, H = 1280, 720
FPS = 30
DURATION_S = 65                 # > 1 minuto, como pede o enunciado
SCALE_PX_PER_MM = 4.0           # mundo sintético: 4 px = 1 mm
PENDRIVE_LENGTH_MM = 55.0       # referência: pendrive de 5,5 cm
PENDRIVE_WIDTH_MM = 20.0

# (comprimento_mm, largura_mm, cor BGR) — algumas peças propositalmente
# fora das tolerâncias do config.yaml para gerar REPROVADO
PIECES = [
    (80.0, 40.0, (60, 90, 200)),    # peca_grande OK
    (50.0, 50.0, (80, 170, 90)),    # peca_media OK
    (25.0, 25.0, (200, 140, 60)),   # peca_pequena OK
    (85.0, 40.0, (60, 60, 220)),    # peca_grande DEFEITUOSA (comprimento)
    (50.0, 45.0, (100, 190, 110)),  # peca_media DEFEITUOSA (largura)
    (25.5, 24.5, (220, 160, 80)),   # peca_pequena OK (dentro da tolerância)
]


def main() -> None:
    os.makedirs("data/videos", exist_ok=True)
    out_path = "data/videos/sintetico.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    n_frames = FPS * DURATION_S
    spacing = 700                   # px entre os centros das peças
    speed = 4.0                     # px/frame (todas as 6 peças cruzam a cena em 65s)

    rng = np.random.default_rng(42)

    for f in range(n_frames):
        frame = np.full((H, W, 3), 235, np.uint8)          # bancada clara
        noise = rng.normal(0, 4, (H, W, 3))                # ruído do sensor
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        # pendrive de referência fixo no canto inferior esquerdo
        pd_l = int(PENDRIVE_LENGTH_MM * SCALE_PX_PER_MM)
        pd_w = int(PENDRIVE_WIDTH_MM * SCALE_PX_PER_MM)
        pd_cx, pd_cy = 160, H - 90
        cv2.rectangle(
            frame,
            (pd_cx - pd_l // 2, pd_cy - pd_w // 2),
            (pd_cx + pd_l // 2, pd_cy + pd_w // 2),
            (90, 60, 40),
            -1,
        )
        # "conector" metálico para parecer um pendrive de verdade
        cv2.rectangle(
            frame,
            (pd_cx + pd_l // 2 - 30, pd_cy - pd_w // 2 + 8),
            (pd_cx + pd_l // 2 - 5, pd_cy + pd_w // 2 - 8),
            (180, 180, 190),
            -1,
        )

        # peças atravessando a esteira
        for i, (len_mm, wid_mm, color) in enumerate(PIECES):
            x = int(W + i * spacing - f * speed)
            if x < -300 or x > W + 300:
                continue
            y = H // 2 + 60
            l_px = int(len_mm * SCALE_PX_PER_MM)
            w_px = int(wid_mm * SCALE_PX_PER_MM)
            angle = 12.0 * (i % 3 - 1)                      # leve rotação
            rect = ((x, y), (l_px, w_px), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(frame, [box], color)
            cv2.polylines(frame, [box], True, tuple(int(c * 0.6) for c in color), 2)

        writer.write(frame)

    writer.release()
    print(f"Vídeo sintético gerado: {out_path} ({DURATION_S}s @ {FPS}fps)")

    # ground truth: os track_ids seguem a ordem de entrada na cena (1, 2, ...)
    gt_path = "data/ground_truth_sintetico.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["track_id", "comprimento_real_mm", "largura_real_mm"])
        for i, (len_mm, wid_mm, _) in enumerate(PIECES, start=1):
            wr.writerow([i, len_mm, wid_mm])
    print(f"Ground truth gerado:    {gt_path}")


if __name__ == "__main__":
    main()
