# Sistema de Inspeção de Qualidade com SAM

Sistema automatizado de inspeção que processa vídeo de uma esteira/bancada,
segmenta peças com **FastSAM** (variante em tempo real do Segment Anything
Model), rastreia cada peça e calcula **medições geométricas reais**
(comprimento, largura e área em mm) para classificação Aprovado/Reprovado.

## Estrutura

```
├── main.py                     # CLI: calibrate | run | report
├── config.yaml                 # parâmetros (calibração, SAM, tolerâncias)
├── requirements.txt
├── src/
│   ├── preprocessing.py        # Fase 1: filtros Gaussiano/média
│   ├── calibration.py          # Fase 1: referência -> fator mm/px
│   ├── segmentation.py         # Fase 2: FastSAM + prompts automáticos
│   ├── tracking.py             # Fase 2: rastreamento por centroide
│   ├── metrology.py            # Fase 3: minAreaRect, área, tolerâncias
│   ├── visualization.py        # desenho das anotações
│   ├── report.py               # consolidação + MAE/MSE
│   └── pipeline.py             # orquestração de tudo
├── tools/
│   └── generate_test_video.py  # vídeo sintético p/ validar o pipeline
└── data/
    ├── videos/                 # coloque o vídeo real aqui
    ├── output/                 # vídeo anotado + medicoes.csv
    └── ground_truth.csv        # medições manuais (paquímetro)
```

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# GPU NVIDIA (opcional, mais rápido):
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Na primeira execução o peso `FastSAM-s.pt` (~23 MB) é baixado automaticamente.

## Uso

```powershell
# 1. (opcional) gerar vídeo sintético de teste
.venv\Scripts\python tools\generate_test_video.py

# 2. verificar a calibração (fator mm/px)
.venv\Scripts\python main.py calibrate --video data\videos\sintetico.mp4

# 3. pipeline completo -> data/output/video_anotado.mp4 + medicoes.csv
.venv\Scripts\python main.py run --video data\videos\sintetico.mp4

# 4. análise de erro (MAE/MSE) contra medições de paquímetro
.venv\Scripts\python main.py report --ground-truth data\ground_truth_sintetico.csv
```

Quando o **vídeo real** estiver disponível: coloque-o em `data/videos/`,
ajuste `config.yaml` (tipo/dimensão da referência, tolerâncias das classes)
e rode os mesmos comandos apontando para ele.

## Mapeamento para o enunciado

| Requisito | Onde está |
|---|---|
| Filtros digitais (média/Gaussiano) | `src/preprocessing.py` |
| Fator de escala mm/px (pendrive/retângulo, moeda ou xadrez) | `src/calibration.py` |
| SAM com prompts automáticos | `src/segmentation.py` (modos `everything` e `auto_points`) |
| Rastreamento | `src/tracking.py` |
| `cv2.minAreaRect` -> comprimento/largura | `src/metrology.py` |
| Área da máscara em mm² | `src/metrology.py` |
| Tolerância -> Aprovado/Reprovado | `src/metrology.py` + `config.yaml` |
| Tabela real vs. medido, MAE/MSE | `src/report.py` (`main.py report`) |

## Ground truth (medições com paquímetro)

Preencha `data/ground_truth.csv` com uma linha por peça:

```csv
track_id,comprimento_real_mm,largura_real_mm
1,80.0,40.0
2,50.1,49.9
```

O `track_id` é o número mostrado no vídeo anotado (`#1`, `#2`, ...).
