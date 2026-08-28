"""Calibração da câmera: cálculo da relação pixel/milímetro (Fase 1).

Detecta automaticamente o objeto de referência (moeda ou padrão xadrez),
mede sua dimensão em pixels e calcula:

    fator_de_escala (mm/px) = dimensão_real_mm / dimensão_em_pixels

A calibração é feita sobre vários frames e o resultado final é a mediana,
o que a torna robusta a falhas de detecção pontuais.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    scale_mm_per_px: float          # fator de conversão mm/pixel
    reference_center: tuple | None  # centro (x, y) da referência no frame
    reference_radius_px: float | None  # raio em px (se moeda)
    n_samples: int                  # nº de frames em que a referência foi detectada
    reference_end_frame: int | None = None  # último frame da referência móvel (modo first_object)


def detect_circle_reference(
    gray: np.ndarray, min_radius_px: int, max_radius_px: int
) -> tuple[tuple[int, int], float] | None:
    """Detecta uma referência circular (ex.: moeda) via Transformada de Hough.

    Returns:
        ((cx, cy), raio_px) ou None se nada for detectado.
    """
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(gray.shape) // 4,
        param1=120,
        param2=50,
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
    )
    if circles is None:
        return None
    # HoughCircles ordena por acumulador: o primeiro é o mais confiável
    cx, cy, r = circles[0][0]
    return (int(cx), int(cy)), float(r)


def detect_rect_reference(
    gray: np.ndarray, roi_px: tuple[int, int, int, int], min_area_px: float = 500
) -> tuple[tuple[int, int], float, float] | None:
    """Detecta uma referência retangular (ex.: pendrive) dentro de uma ROI.

    Limiariza a região (Otsu), pega o maior contorno e mede o lado maior do
    retângulo mínimo.

    Args:
        roi_px: (x0, y0, x1, y1) da região onde a referência aparece.

    Returns:
        ((cx, cy) no frame completo, comprimento_px, meia_diagonal_px)
        ou None se nada for detectado.
    """
    x0, y0, x1, y1 = roi_px
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # o objeto deve ser a minoria dos pixels da ROI
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None

    rect = cv2.minAreaRect(largest)
    (w_px, h_px) = rect[1]
    if w_px == 0 or h_px == 0:
        return None
    center = (int(rect[0][0]) + x0, int(rect[0][1]) + y0)
    length_px = float(max(w_px, h_px))
    half_diag_px = float(np.hypot(w_px, h_px) / 2)
    return center, length_px, half_diag_px


def detect_rect_reference_sam(
    frame: np.ndarray,
    roi_px: tuple[int, int, int, int],
    segmenter,
    min_area_px: float = 500,
) -> tuple[tuple[int, int], float, float] | None:
    """Detecta a referência retangular usando o próprio FastSAM na ROI.

    Mais robusto que o Otsu em cenas com pouco contraste ou iluminação
    irregular: segmenta a ROI, descarta máscaras que tocam as bordas
    (fundo, piso) e escolhe o maior objeto restante.

    Args:
        segmenter: instância de ``SamSegmenter`` (usa ``.model``/``.device``).

    Returns:
        ((cx, cy) no frame completo, comprimento_px, meia_diagonal_px) ou None.
    """
    x0, y0, x1, y1 = roi_px
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]

    results = segmenter.model(
        crop,
        device=segmenter.device,
        imgsz=512,
        conf=0.4,
        iou=0.9,
        retina_masks=True,
        verbose=False,
    )
    r = results[0]
    if r.masks is None:
        return None

    best_mask, best_area = None, 0.0
    margin = 3
    for mask_t in r.masks.data:
        mask = (mask_t.cpu().numpy() > 0.5).astype(np.uint8) * 255
        if mask.shape != (ch, cw):
            mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        area = float(cv2.countNonZero(mask))
        if area < min_area_px or area > 0.3 * ch * cw:
            continue
        ys, xs = np.nonzero(mask)
        if (
            xs.min() <= margin
            or ys.min() <= margin
            or xs.max() >= cw - 1 - margin
            or ys.max() >= ch - 1 - margin
        ):
            continue
        if area > best_area:
            best_mask, best_area = mask, area

    if best_mask is None:
        return None

    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (w_px, h_px) = rect[1]
    if w_px == 0 or h_px == 0:
        return None
    center = (int(rect[0][0]) + x0, int(rect[0][1]) + y0)
    return center, float(max(w_px, h_px)), float(np.hypot(w_px, h_px) / 2)


def calibrate_first_object(
    video_path: str,
    cfg: dict,
    segmenter,
    tracking_cfg: dict | None = None,
    preprocessing_cfg: dict | None = None,
    frame_stride: int = 1,
) -> CalibrationResult:
    """Calibra usando o PRIMEIRO objeto que atravessa a cena (esteira).

    A referência é móvel: entra por uma borda, atravessa e sai. O FastSAM
    segmenta cada frame, um tracker local acompanha o primeiro objeto válido
    e a maior dimensão dele (``minAreaRect``) é medida em todos os frames em
    que aparece; a escala vem da mediana dessas medições:

        escala (mm/px) = known_length_mm / mediana(comprimento_px)

    Também devolve ``reference_end_frame`` (último frame em que a referência
    foi vista + margem) para o pipeline marcar como REF os tracks iniciados
    até esse ponto.

    Args:
        cfg: bloco ``calibration`` do config.yaml.
        segmenter: instância de ``SamSegmenter`` (obrigatório neste modo).
        tracking_cfg: bloco ``tracking`` do config (opcional).
        preprocessing_cfg: bloco ``preprocessing`` do config (opcional).
        frame_stride: mesmo stride usado no pipeline, por consistência.
    """
    from .metrology import measure_mask
    from .preprocessing import apply_filter
    from .tracking import CentroidTracker

    ref = cfg["reference"]
    known_mm = float(ref["known_length_mm"])
    min_track_frames = int(ref.get("min_track_frames", 5))
    min_length_px = float(ref.get("min_length_px", 40))
    max_length_px = float(ref.get("max_length_px", float("inf")))
    max_scan_frames = int(ref.get("max_scan_frames", 900))

    tcfg = tracking_cfg or {}
    tracker = CentroidTracker(
        max_distance_px=float(tcfg.get("max_distance_px", 100)),
        max_missed_frames=int(tcfg.get("max_missed_frames", 15)),
        min_hits=int(tcfg.get("min_hits", 3)),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    lengths: dict[int, list[float]] = {}
    first_frame: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    reference_tid: int | None = None

    frame_idx = 0
    while frame_idx < max_scan_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        if preprocessing_cfg:
            frame = apply_filter(
                frame,
                preprocessing_cfg.get("filter", "gaussian"),
                int(preprocessing_cfg.get("kernel_size", 5)),
            )

        objects = segmenter.segment(frame)
        assignment = tracker.update([o.centroid for o in objects])

        for j, obj in enumerate(objects):
            m = measure_mask(obj.mask, scale_mm_per_px=1.0)  # 1.0 -> medidas em px
            if m is None:
                continue
            tid = assignment[j]
            lengths.setdefault(tid, []).append(m.length_mm)
            first_frame.setdefault(tid, frame_idx)
            last_seen[tid] = frame_idx

        # elege como referência o track mais antigo que ficou consistente
        if reference_tid is None:
            for tid in sorted(lengths, key=lambda t: first_frame[t]):
                samples = lengths[tid]
                if (
                    len(samples) >= min_track_frames
                    and min_length_px <= np.median(samples) <= max_length_px
                ):
                    reference_tid = tid
                    break

        # encerra quando a referência sai de cena (track deixou de estar ativo)
        if reference_tid is not None and reference_tid not in tracker.active:
            break

        frame_idx += 1

    cap.release()

    if reference_tid is None:
        raise RuntimeError(
            "Nenhum objeto de referência válido passou nos frames iniciais. "
            "Ajuste 'min_length_px'/'max_scan_frames' na calibração ou "
            "'min_area_px'/'conf' na segmentação."
        )

    samples = np.array(lengths[reference_tid], dtype=float)
    median_px = float(np.median(samples))
    if median_px > 0 and float(np.std(samples)) / median_px > 0.15:
        print(
            "      AVISO: medições da referência instáveis "
            f"(desvio {np.std(samples):.1f} px sobre mediana {median_px:.1f} px)"
        )

    return CalibrationResult(
        scale_mm_per_px=known_mm / median_px,
        reference_center=None,
        reference_radius_px=None,
        n_samples=len(samples),
        reference_end_frame=last_seen[reference_tid] + 2 * frame_stride,
    )


def detect_chessboard_reference(
    gray: np.ndarray, cols: int, rows: int
) -> float | None:
    """Detecta um padrão xadrez e retorna a distância média entre cantos
    adjacentes, em pixels (equivale ao lado de um quadrado do padrão)."""
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    grid = corners.reshape(rows, cols, 2)
    # distâncias horizontais e verticais entre cantos vizinhos
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    return float(np.median(np.concatenate([dx.ravel(), dy.ravel()])))


def calibrate_from_video(
    video_path: str,
    cfg: dict,
    segmenter=None,
    tracking_cfg: dict | None = None,
    preprocessing_cfg: dict | None = None,
    frame_stride: int = 1,
) -> CalibrationResult:
    """Percorre os primeiros frames do vídeo e calcula o fator de escala.

    Args:
        video_path: caminho do vídeo.
        cfg: bloco ``calibration`` do config.yaml.
        segmenter: ``SamSegmenter`` opcional; obrigatório no tipo
            ``first_object`` e usado no tipo ``rect`` com ``method: sam``.
        tracking_cfg/preprocessing_cfg/frame_stride: usados apenas pelo tipo
            ``first_object`` (referência móvel na esteira).
    """
    ref = cfg["reference"]

    if ref["type"] == "first_object":
        if segmenter is None:
            raise ValueError("calibração 'first_object' requer o FastSAM (segmenter)")
        return calibrate_first_object(
            video_path,
            cfg,
            segmenter,
            tracking_cfg=tracking_cfg,
            preprocessing_cfg=preprocessing_cfg,
            frame_stride=frame_stride,
        )

    n_frames = int(cfg.get("n_frames", 15))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    scales: list[float] = []
    centers: list[tuple[int, int]] = []
    radii: list[float] = []

    read_frames = 0
    while len(scales) < n_frames and read_frames < n_frames * 10:
        ok, frame = cap.read()
        if not ok:
            break
        read_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ref["type"] == "circle":
            det = detect_circle_reference(
                gray, int(ref["min_radius_px"]), int(ref["max_radius_px"])
            )
            if det is None:
                continue
            center, radius = det
            diameter_px = 2.0 * radius
            scales.append(float(ref["known_diameter_mm"]) / diameter_px)
            centers.append(center)
            radii.append(radius)
        elif ref["type"] == "rect":
            h, w = gray.shape
            fx0, fy0, fx1, fy1 = ref.get("roi", [0.0, 0.5, 0.5, 1.0])
            roi_px = (int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h))
            min_area = float(ref.get("min_area_px", 500))
            if ref.get("method", "sam") == "sam" and segmenter is not None:
                det = detect_rect_reference_sam(frame, roi_px, segmenter, min_area)
            else:
                det = detect_rect_reference(gray, roi_px, min_area)
            if det is None:
                continue
            center, length_px, half_diag_px = det
            scales.append(float(ref["known_length_mm"]) / length_px)
            centers.append(center)
            radii.append(half_diag_px)
        elif ref["type"] == "chessboard":
            cb = ref["chessboard"]
            square_px = detect_chessboard_reference(gray, int(cb["cols"]), int(cb["rows"]))
            if square_px is None:
                continue
            scales.append(float(cb["square_size_mm"]) / square_px)
        else:
            raise ValueError(f"Tipo de referência desconhecido: {ref['type']!r}")

    cap.release()

    if not scales:
        raise RuntimeError(
            "Objeto de referência não encontrado nos frames iniciais. "
            "Ajuste 'calibration' no config.yaml (tipo, faixa de raio, etc.)."
        )

    center = None
    radius = None
    if centers:
        center = tuple(int(v) for v in np.median(np.array(centers), axis=0))
        radius = float(np.median(radii))

    return CalibrationResult(
        scale_mm_per_px=float(np.median(scales)),
        reference_center=center,
        reference_radius_px=radius,
        n_samples=len(scales),
    )
