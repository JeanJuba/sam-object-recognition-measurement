"""Segmentação com FastSAM (Fase 2).

Integra o FastSAM (variante do SAM otimizada para tempo real, via
``ultralytics``) ao pipeline de vídeo, com duas estratégias de prompting
automático:

* ``everything``  – o modelo segmenta tudo e as máscaras são filtradas por
  área, posição e sobreposição com a referência de calibração;
* ``auto_points`` – candidatos são detectados por limiarização/contornos
  (OpenCV) e seus centroides viram *point prompts* para o FastSAM.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentedObject:
    mask: np.ndarray        # máscara binária (uint8, 0/255) no tamanho do frame
    centroid: tuple[int, int]
    area_px: float


class SamSegmenter:
    def __init__(self, cfg: dict):
        from ultralytics import FastSAM  # import tardio: torch é pesado

        self.cfg = cfg
        self.model = FastSAM(cfg.get("model", "FastSAM-s.pt"))
        device = cfg.get("device", "auto")
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    # ------------------------------------------------------------------
    # Geração automática de prompts (estratégia auto_points)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_point_prompts(frame: np.ndarray, min_area_px: float) -> list[list[int]]:
        """Gera pontos de prompt a partir de contornos salientes da cena.

        Usa Otsu sobre o canal de luminância para separar objetos do fundo
        e devolve o centroide de cada blob relevante.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # garante objetos claros sobre fundo escuro ou o inverso: usa a
        # versão em que o fundo (maioria dos pixels) é 0
        if np.count_nonzero(binary) > binary.size / 2:
            binary = cv2.bitwise_not(binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for c in contours:
            if cv2.contourArea(c) < min_area_px:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            points.append([int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])])
        return points

    # ------------------------------------------------------------------
    # Refinamento da máscara: mantém só a face superior (mais clara)
    # ------------------------------------------------------------------
    @staticmethod
    def refine_top_face(gray: np.ndarray, mask: np.ndarray, cfg: dict) -> list[np.ndarray]:
        """Remove da máscara a face lateral sombreada do objeto.

        Com a câmera angulada, objetos com altura (ex.: borrachas) mostram a
        lateral além da face superior, inflando área e contorno. Um limiar de
        Otsu sobre o brilho dos pixels da máscara separa a face superior
        (clara) da lateral (escura); ficam as componentes claras relevantes,
        com os buracos internos preenchidos.

        Devolve uma lista de máscaras: peças encostadas que o SAM funde numa
        máscara só (ex.: par de borrachas) têm faces superiores separadas
        pelo vão sombreado e viram objetos individuais.

        Salvaguardas — devolve ``[máscara original]`` quando:
        * o contraste topo/lateral é baixo (objeto sem lateral visível);
        * a parte mantida ficaria pequena demais (limiar cortou o objeto);
        * quase nada foi removido (refino irrelevante).
        """
        vals = gray[mask > 0]
        if vals.size < 50:
            return [mask]
        thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bright_vals = vals[vals >= thr]
        dark_vals = vals[vals < thr]
        if bright_vals.size == 0 or dark_vals.size == 0:
            return [mask]
        contrast = float(bright_vals.mean()) - float(dark_vals.mean())
        if contrast < float(cfg.get("min_contrast", 25)):
            return [mask]

        bright = np.zeros_like(mask)
        bright[(mask > 0) & (gray >= thr)] = 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [mask]

        # mantém toda componente clara relevante (>= fração da maior);
        # o contorno externo é preenchido para que manchas escuras dentro
        # da face superior (sujeira, textura) não virem buracos
        areas = [cv2.contourArea(c) for c in contours]
        largest = max(areas)
        min_component = largest * float(cfg.get("min_component_ratio", 0.3))
        refined_masks = []
        kept_px = 0
        for contour, area in zip(contours, areas):
            if area < min_component:
                continue
            refined = np.zeros_like(mask)
            cv2.drawContours(refined, [contour], -1, 255, cv2.FILLED)
            refined_masks.append(refined)
            kept_px += cv2.countNonZero(refined)

        keep_ratio = kept_px / max(cv2.countNonZero(mask), 1)
        if keep_ratio < float(cfg.get("min_keep_ratio", 0.3)):
            return [mask]
        if keep_ratio > float(cfg.get("max_keep_ratio", 0.95)):
            return [mask]
        return refined_masks

    # ------------------------------------------------------------------
    # Segmentação de um frame
    # ------------------------------------------------------------------
    def segment(
        self,
        frame: np.ndarray,
        exclude_center: tuple[int, int] | None = None,
        exclude_radius_px: float | None = None,
    ) -> list[SegmentedObject]:
        """Segmenta o frame e devolve os objetos de interesse.

        Args:
            frame: imagem BGR (já pré-processada).
            exclude_center/exclude_radius_px: região do objeto de referência
                de calibração, que não deve ser tratada como peça.
        """
        cfg = self.cfg
        kwargs = dict(
            device=self.device,
            imgsz=int(cfg.get("imgsz", 1024)),
            conf=float(cfg.get("conf", 0.4)),
            iou=float(cfg.get("iou", 0.9)),
            retina_masks=True,
            verbose=False,
        )

        if cfg.get("prompt_mode", "everything") == "auto_points":
            points = self.generate_point_prompts(frame, float(cfg.get("min_area_px", 1500)))
            if not points:
                return []
            results = self.model(frame, points=points, labels=[1] * len(points), **kwargs)
        else:
            results = self.model(frame, **kwargs)

        r = results[0]
        if r.masks is None:
            return []

        h, w = frame.shape[:2]
        min_area = float(cfg.get("min_area_px", 1500))
        max_area = float(cfg.get("max_area_ratio", 0.35)) * h * w
        margin = int(cfg.get("border_margin_px", 5))
        max_aspect = float(cfg.get("max_aspect_ratio", 0))  # 0 = desativado

        refine_cfg = cfg.get("refine_top_face") or {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if refine_cfg.get("enabled") else None

        candidates: list[np.ndarray] = []
        for mask_t in r.masks.data:
            mask = (mask_t.cpu().numpy() > 0.5).astype(np.uint8) * 255
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            area = float(cv2.countNonZero(mask))
            if area < min_area or area > max_area:
                continue
            candidates.append(mask)

        # dedup ANTES do refino: variantes da mesma peça (objeto, objeto+sombra)
        # se sobrepõem quase totalmente agora, mas refinadas encolhem para
        # fragmentos claros distintos e escapariam da supressão final
        candidates = self._dedup_masks(candidates)

        if gray is not None:
            refined: list[np.ndarray] = []
            for mask in candidates:
                refined.extend(self.refine_top_face(gray, mask, refine_cfg))
            candidates = refined

        objects: list[SegmentedObject] = []
        for mask in candidates:
            area = float(cv2.countNonZero(mask))
            if area < min_area or area > max_area:
                continue

            ys, xs = np.nonzero(mask)
            # descarta máscaras encostadas na borda (objetos cortados)
            if (
                xs.min() <= margin
                or ys.min() <= margin
                or xs.max() >= w - 1 - margin
                or ys.max() >= h - 1 - margin
            ):
                continue

            # descarta máscaras extremamente alongadas (riscos/costuras da esteira)
            if max_aspect > 0:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    rw, rh = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
                    if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) > max_aspect:
                        continue

            cx, cy = int(xs.mean()), int(ys.mean())

            # descarta a máscara do objeto de referência de calibração
            if exclude_center is not None and exclude_radius_px is not None:
                dist = np.hypot(cx - exclude_center[0], cy - exclude_center[1])
                if dist < exclude_radius_px * 1.5:
                    continue

            objects.append(SegmentedObject(mask=mask, centroid=(cx, cy), area_px=area))

        # remove máscaras duplicadas/aninhadas (o modo everything às vezes
        # devolve o mesmo objeto mais de uma vez)
        return self._suppress_duplicates(objects)

    @staticmethod
    def _dedup_masks(masks: list[np.ndarray], iou_thr: float = 0.7) -> list[np.ndarray]:
        """Remove máscaras duplicadas/aninhadas, mantendo sempre a maior."""
        masks = sorted(masks, key=cv2.countNonZero, reverse=True)
        kept: list[np.ndarray] = []
        for mask in masks:
            area = float(cv2.countNonZero(mask))
            duplicate = False
            for k in kept:
                inter = cv2.countNonZero(cv2.bitwise_and(mask, k))
                union = cv2.countNonZero(cv2.bitwise_or(mask, k))
                if union > 0 and inter / union > iou_thr:
                    duplicate = True
                    break
                if inter / max(area, 1.0) > 0.85:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(mask)
        return kept

    @staticmethod
    def _suppress_duplicates(objects: list[SegmentedObject], iou_thr: float = 0.7) -> list[SegmentedObject]:
        objects = sorted(objects, key=lambda o: o.area_px, reverse=True)
        kept: list[SegmentedObject] = []
        for obj in objects:
            duplicate = False
            for k in kept:
                inter = cv2.countNonZero(cv2.bitwise_and(obj.mask, k.mask))
                union = cv2.countNonZero(cv2.bitwise_or(obj.mask, k.mask))
                if union > 0 and inter / union > iou_thr:
                    duplicate = True
                    break
                # máscara quase totalmente contida em outra maior
                if inter / max(obj.area_px, 1.0) > 0.85:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(obj)
        return kept
