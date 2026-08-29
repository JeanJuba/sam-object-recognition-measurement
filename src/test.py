import os

import cv2
import numpy as np
from ultralytics import FastSAM

model = FastSAM('FastSAM-s.pt')  # baixa automaticamente na primeira vez

folder = './data/images/'
images = ['lapis_pendrive_controle.png', 'ferramentas.jpeg']

os.makedirs('./data/output/', exist_ok=True)

MAX_AREA_FRAC = 0.5  # descarta mascaras maiores que 50% da imagem (fundo/mesa)

for idx, img_name in enumerate(images):
    path = os.path.join(folder, img_name)
    results = model(path, device='cuda', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    result = results[0]

    img = result.orig_img.copy()
    h, w = img.shape[:2]

    if result.masks is not None:
        rng = np.random.default_rng(42)
        for mask in result.masks.data.cpu().numpy():
            mask = (mask * 255).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if (mask > 0).sum() > MAX_AREA_FRAC * h * w:
                continue  # mascara de fundo
            color = rng.integers(60, 255, size=3).tolist()
            overlay = img.copy()
            overlay[mask > 0] = color
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 2)

    cv2.imwrite(os.path.join('./data/output/', f'output_{idx}.jpg'), img)
