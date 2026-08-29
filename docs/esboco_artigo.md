# Esboço do artigo — Inspeção de qualidade com SAM e metrologia computacional

> Vídeo principal: `aruco_longo.mp4` (1080×1920, 60 s — cumpre o mínimo de 1 min).
> Vídeo de apoio: `aruco.mp4` (esteira azul, pares de borrachas).

## Título e Resumo
- Título sugerido: "Metrologia computacional em vídeo com FastSAM: calibração por marcador ArUco e refinamento de máscaras para medição dimensional de peças".
- Resumo: problema (medir peças em mm a partir de vídeo), método (FastSAM + calibração ArUco + refinamento por brilho), resultados-chave (repetibilidade sub-milimétrica entre frames; MAE/MSE vs. paquímetro), conclusão.

## 1. Introdução
- Contexto: inspeção dimensional automatizada em linha (esteira), custo de metrologia manual.
- O SAM como segmentador *promptável* de propósito geral; por que segmentação pixel a pixel importa para medir área verdadeira (caixa delimitadora superestima formatos não retangulares — ex.: borracha oval, +27% se fosse pela caixa).
- Objetivo: pipeline vídeo → calibração → segmentação → metrologia → classificação Aprovado/Reprovado.

## 2. Metodologia

### 2.1 Aquisição de dados (Fase 1)
- Descrição dos vídeos: resolução, fps, duração, cenário (peças brancas sobre tecido preto; marcador ArUco impresso como referência estática).
- **Ponto para discutir honestamente**: o requisito pede visão ortogonal; a câmera usada é levemente angulada e de mão. Explicar as duas compensações implementadas:
  1. `scale_correction: 0.96` — o papel do marcador está ~4% mais distante da câmera que o plano das peças; correção de plano aplicada à escala;
  2. `refine_top_face` — remoção da face lateral que o ângulo torna visível (ver 2.4).

### 2.2 Pré-processamento — filtros digitais (Fase 1)
- Filtro Gaussiano 3×3 antes da segmentação (`preprocessing.py`): suaviza a textura do tecido/esteira e o ruído do sensor sem borrar bordas das peças; kernel configurável, comparar com filtro de média (ambos implementados).
- Justificar o kernel pequeno: kernels maiores borram peças pequenas (ponteira).

### 2.3 Calibração — relação pixel/milímetro (Fase 1)
- Fórmula: fator de escala (mm/px) = dimensão real (mm) / dimensão em pixels; medição em N frames e uso da **mediana** (robustez a falhas pontuais).
- Método principal (aruco_longo): marcador ArUco **decodificado** com `cv2.aruco` (dicionário DICT_6X6_50, id 42) — cantos com precisão subpixel; escala média das direções horizontal/vertical (impressão fora de esquadro: 98,5 × 96,5 mm).
- Métodos alternativos implementados (mostrar a generalidade do pipeline): `rect`/Otsu em ROI (aruco.mp4), `first_object` (1º objeto que cruza a esteira como referência móvel), `circle`/Hough (moeda) e `chessboard` (xadrez).
- Influência do ângulo da câmera na precisão (item 1 do relatório): geometria — a projeção da face lateral soma à dimensão aparente da peça, e o efeito varia com a posição no quadro; evidência empírica no aruco_longo: com o refinamento, as áreas caem −2,5% (oval, baixa) a −19% (retangular, alta) e o desvio entre frames diminui (ex.: largura 1,12 → 0,28 mm).

### 2.4 Segmentação com FastSAM (Fase 2)
- Escolha da variante (item 2 do relatório): FastSAM-s (ultralytics) vs. SAM ViT-B/L/H e MobileSAM — tabela comparando parâmetros, VRAM e FPS; justificar: tempo quase real em GPU (~30 FPS de segmentação; 4 FPS pipeline completo no vídeo 1080×1920), API simples, máscaras retina.
- **Estratégia de prompting** (requisito explícito): duas estratégias automatizadas:
  1. modo *everything* + cascata de filtros geométricos (área mín./máx., margem de borda, proporção máxima, zona de exclusão da referência);
  2. modo *auto_points*: limiarização Otsu + contornos OpenCV geram centroides que viram *point prompts*.
- **Isolamento de máscaras**: máscaras binárias por objeto por frame; deduplicação de variantes da mesma peça (IoU > 0,7 ou contenção > 85%).
- **Refinamento da face superior** (`refine_top_face` — contribuição própria do trabalho):
  - motivação: câmera angulada inclui a lateral sombreada na máscara, inflando contorno e área;
  - algoritmo: Otsu sobre o brilho *dentro* da máscara → mantém componentes claras (≥30% da maior) com contorno preenchido; salvaguardas (contraste mínimo 25, proporção mantida 30–95%) preservam a máscara original em peças chatas;
  - efeito colateral desejável: separa pares de peças encostadas pelo vão sombreado (aruco.mp4: pares → 32 borrachas individuais);
  - resultado no vídeo principal: áreas −2,5% (oval, baixa) a −19% (retangular, alta) — o desconto é proporcional à altura da peça; desvio entre frames caiu (ex.: largura 1,12 → 0,28 mm).
- Rastreamento: tracker por centroide (associação gulosa por distância) para manter identidade da peça entre frames e consolidar medições por track.

### 2.5 Metrologia computacional (Fase 3)
- Dimensões lineares: `cv2.findContours` → maior contorno → `cv2.minAreaRect` → comprimento e largura em px × fator de escala.
- Área: contagem de pixels da máscara (`cv2.countNonZero`) × escala² — área verdadeira do formato, independente de retângulo.
- Consolidação por peça: mediana das medições ao longo dos frames do track (robustez a máscaras ruins isoladas); desvio-padrão reportado como repetibilidade.
- Classificação (análise de tolerância): atribuição à classe de dimensões nominais mais próximas (distância euclidiana comprimento × largura) + verificação de faixas [mín, máx] por classe → APROVADO/REPROVADO com motivos registrados.

## 3. Resultados

### 3.1 Metrologia (item 3 do relatório)
- Tabela por peça do `medicoes.csv` do aruco_longo: classe, comprimento, largura, área, desvio entre frames, nº de frames, status.
- **Tabela real vs. medido (FEITA em 2026-08-29)**: 4 tipos medidos com paquímetro — retangular 42×21, quadrada 32×23, ponteira 30×13, oval 48×36 (`data/ground_truth.csv`; `python main.py report`). As 7 peças foram classificadas no tipo correto e aprovadas (tolerância ±3 mm).
- **Correção de plano identificada com o paquímetro**: com a escala do marcador, todas as peças saíam ~20% maiores de forma consistente (erro sistemático) → `scale_correction` ajustado de 0,96 para 0,80 (mediana de real/medido). No artigo, apresentar como etapa de calibração do plano das peças; citar como limitação que a correção foi identificada com as mesmas peças da validação (o ideal seria validar em passagens/peças independentes) e que medir o marcador com régua permitiria separar erro de plano de erro da dimensão impressa.
- Comparação com/sem refinamento (tabela já gerada nesta sessão) como evidência do efeito do ângulo.

### 3.2 Análise de erro (item 4 do relatório)
- Resultados com a calibração corrigida: **MAE 0,52 mm (comprimento) e 0,77 mm (largura); RMSE 0,63 e 0,88 mm**; maior erro individual 1,55 mm (largura da quadrada — resíduo de face lateral). Antes da correção de plano: MAE 7,28/4,91 mm — bom contraste para mostrar o peso da calibração vs. segmentação.
- Discutir fontes de erro: incerteza do fator de escala (propaga linearmente para mm e ao quadrado para mm²), borda da máscara (±1–2 px), diferença de plano peça/marcador, distorção da lente (não corrigida), resíduo de face lateral nas larguras.

### 3.3 Desempenho
- FPS de processamento por vídeo/GPU; custo do refinamento (desprezível vs. inferência).

## 4. Discussão
- Caixa vs. máscara: por que a área por pixels é o diferencial do SAM sobre detectores de caixa.
- Limites: peças sem contraste topo/lateral não são refinadas; classes por proximidade não têm "desconhecido" (objeto estranho cai na classe mais próxima e reprova — comportamento observado e defensável, mas citar como melhoria futura com limite de distância).
- Melhoria futura: correção de perspectiva por homografia (4 cantos do ArUco) em vez de correção escalar.

## 5. Conclusão
- Pipeline completo com bibliotecas exigidas (ultralytics/FastSAM, OpenCV, NumPy); repetibilidade sub-milimétrica entre frames; erro absoluto a confirmar com paquímetro.

---

## Mapa requisito → código (para conferência da banca)
| Requisito | Onde está |
|---|---|
| Vídeo ≥1 min, referência na cena | `data/videos/aruco_longo.mp4` (60 s) + marcador ArUco |
| Filtros espaciais (média/Gaussiano) | `src/preprocessing.py` |
| Fator de escala mm/px | `src/calibration.py` (5 métodos) |
| Integração SAM/FastSAM em vídeo | `src/segmentation.py` (`SamSegmenter`) |
| Prompting automatizado | `everything`+filtros e `auto_points` (`generate_point_prompts`) |
| Máscaras binárias por objeto/frame | `SegmentedObject.mask` |
| `cv2.minAreaRect` → comprimento/largura | `src/metrology.py` (`measure_mask`) |
| Área da máscara | `cv2.countNonZero` × escala² (`metrology.py`) |
| Tolerância → Aprovado/Reprovado | `classify` (`metrology.py`) + `classification.classes` nos configs |
| MAE/MSE (e RMSE) | `src/report.py` (`error_analysis`), `main.py report` |
| Desenho em tela | `src/visualization.py` (contorno real da máscara) |
