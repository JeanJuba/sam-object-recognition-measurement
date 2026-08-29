"""Pipeline principal: vídeo -> calibração -> SAM -> métricas -> saídas."""

import os
import time

import cv2

from .calibration import calibrate_from_video
from .metrology import classify, measure_mask
from .preprocessing import apply_filter
from .report import summarize_tracks
from .segmentation import SamSegmenter
from .tracking import CentroidTracker
from .visualization import draw_calibration, draw_object, draw_reference


def run_pipeline(cfg: dict, video_path: str | None = None, show: bool = False) -> str:
    """Executa o pipeline completo e devolve o caminho do CSV de medições."""
    video_path = video_path or cfg["video"]["input_path"]
    out_dir = cfg["video"].get("output_dir", "data/output")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Fase 2 (modelo): o FastSAM é carregado antes por
    # também ser usado na calibração (method: sam) ----------------
    print("[1/3] Carregando FastSAM...")
    segmenter = SamSegmenter(cfg["segmentation"])
    print(f"      Dispositivo: {segmenter.device}")

    # ---------------- Fase 1: calibração ----------------
    print(f"[2/3] Calibrando com o objeto de referência de '{video_path}'...")
    calib = calibrate_from_video(
        video_path,
        cfg["calibration"],
        segmenter=segmenter,
        tracking_cfg=cfg.get("tracking"),
        preprocessing_cfg=cfg.get("preprocessing"),
        frame_stride=int(cfg["video"].get("frame_stride", 1)),
    )
    print(
        f"      Escala: {calib.scale_mm_per_px:.5f} mm/px "
        f"(mediana de {calib.n_samples} frames)"
    )
    if calib.reference_end_frame is not None:
        print(
            f"      Referência (1º objeto) saiu no frame {calib.reference_end_frame} "
            f"— tracks iniciados até lá serão marcados como REF"
        )

    tcfg = cfg["tracking"]
    tracker = CentroidTracker(
        max_distance_px=float(tcfg["max_distance_px"]),
        max_missed_frames=int(tcfg["max_missed_frames"]),
        min_hits=int(tcfg["min_hits"]),
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = int(cfg["video"].get("frame_stride", 1))

    out_video_path = os.path.join(out_dir, "video_anotado.mp4")
    out_fps = max(fps / stride, 1.0)
    # H.264 preserva texto e linhas finas muito melhor que o mp4v; nem todo
    # OpenCV tem o codec (depende do openh264), então há fallback
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"avc1"), out_fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    pcfg = cfg["preprocessing"]
    classes_cfg = cfg["classification"]["classes"]
    track_first_frame: dict[int, int] = {}  # 1º frame de cada track (p/ marcar REF)
    frame_idx = 0
    processed = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        current_frame = frame_idx
        frame_idx += 1
        processed += 1

        filtered = apply_filter(frame, pcfg.get("filter", "gaussian"), int(pcfg.get("kernel_size", 5)))

        objects = segmenter.segment(
            filtered,
            exclude_center=calib.reference_center,
            exclude_radius_px=calib.reference_radius_px,
        )

        assignment = tracker.update([o.centroid for o in objects])
        for tid in assignment.values():
            track_first_frame.setdefault(tid, current_frame)

        annotated = frame.copy()
        draw_calibration(annotated, calib.reference_center, calib.reference_radius_px, calib.scale_mm_per_px, corners=calib.reference_corners)

        # ---------------- Fase 3: metrologia ----------------
        for j, obj in enumerate(objects):
            m = measure_mask(obj.mask, calib.scale_mm_per_px)
            if m is None:
                continue
            tid = assignment[j]
            # referência móvel (e ruído anterior a ela): marca REF e não mede
            if (
                calib.reference_end_frame is not None
                and track_first_frame[tid] <= calib.reference_end_frame
            ):
                draw_reference(annotated, obj.mask, m, tid)
                continue
            tracker.active[tid].measurements.append(m)
            cls = classify(m.length_mm, m.width_mm, classes_cfg)
            draw_object(annotated, obj.mask, m, cls, tid)

        writer.write(annotated)
        if show:
            cv2.imshow("Inspecao SAM", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if processed % 20 == 0:
            fps_proc = processed / (time.time() - t0)
            print(f"      Frame {frame_idx}/{total} ({fps_proc:.1f} FPS de processamento)")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    # ---------------- Consolidação ----------------
    print("[3/3] Consolidando medições...")
    tracks = tracker.finalize()
    df = summarize_tracks(tracks, classes_cfg)
    csv_path = os.path.join(out_dir, "medicoes.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nPeças detectadas: {len(df)}")
    if not df.empty:
        print(df.to_string(index=False))
    print(f"\nVídeo anotado: {out_video_path}")
    print(f"Medições:      {csv_path}")
    return csv_path
