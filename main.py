"""Sistema de Inspeção de Qualidade Automatizado com SAM.

Uso:
    python main.py calibrate [--video CAMINHO]
        Detecta o objeto de referência e imprime o fator de escala (mm/px).

    python main.py run [--video CAMINHO] [--show]
        Executa o pipeline completo: calibração -> FastSAM -> rastreamento ->
        metrologia -> vídeo anotado + CSV de medições.

    python main.py report [--measurements CSV] [--ground-truth CSV]
        Compara as medições do software com as manuais (paquímetro) e
        calcula MAE / MSE / RMSE.
"""

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeção de qualidade com SAM")
    parser.add_argument("--config", default="config.yaml", help="arquivo de configuração")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="calcula o fator de escala mm/px")
    p_cal.add_argument("--video", default=None)

    p_run = sub.add_parser("run", help="executa o pipeline completo")
    p_run.add_argument("--video", default=None)
    p_run.add_argument("--show", action="store_true", help="exibe o vídeo durante o processamento")

    p_rep = sub.add_parser("report", help="análise de erro (MAE/MSE)")
    p_rep.add_argument("--measurements", default="data/output/medicoes.csv")
    p_rep.add_argument("--ground-truth", default="data/ground_truth.csv")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "calibrate":
        from src.calibration import calibrate_from_video

        video = args.video or cfg["video"]["input_path"]
        segmenter = None
        ref = cfg["calibration"]["reference"]
        if ref["type"] == "rect" and ref.get("method", "sam") == "sam":
            from src.segmentation import SamSegmenter

            segmenter = SamSegmenter(cfg["segmentation"])
        calib = calibrate_from_video(video, cfg["calibration"], segmenter=segmenter)
        print(f"Fator de escala: {calib.scale_mm_per_px:.5f} mm/px")
        print(f"Frames com referência detectada: {calib.n_samples}")
        if calib.reference_center:
            print(f"Centro da referência: {calib.reference_center} | raio: {calib.reference_radius_px:.1f} px")

    elif args.command == "run":
        from src.pipeline import run_pipeline

        run_pipeline(cfg, video_path=args.video, show=args.show)

    elif args.command == "report":
        from src.report import error_analysis

        table, metrics = error_analysis(args.measurements, args.ground_truth)
        print("Tabela comparativa (real vs. medido):\n")
        print(table.to_string(index=False))
        print("\nMétricas de erro:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
