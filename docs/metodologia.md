# Metodologia

Resumo reprodutível do experimento de classificação de sub-técnicas de judô.

## Tarefa
Classificação binária **sutemi_waza vs tachi_waza** (onde `tachi_waza` = `ashi_waza` +
`te_waza` fundidas), a partir de clipes curtos (~4 s) de vídeo.

## Dados
- Clipes de ~4 s, um por golpe, com a classe no nome do arquivo.
- O **estudo** usou o conjunto completo (~1012 clipes, várias fontes, parte privada — ver artigo). Este material público traz um **subconjunto de 594 clipes** de 22 fontes públicas do YouTube (Sutemi 175 / Tachi 419).
- O `source_id` (vídeo-fonte de origem) é derivado do nome
  (`<classe>_<idFonte>_luta<NN>_sub<NN>`), permitindo o split por fonte.
- Vídeos-fonte referenciados por URL (`fontes_videos.csv`); clipes derivados sob solicitação.

## Modelos
- **R(2+1)D-18** e **MC3-18** (arquiteturas de reconhecimento de ação em vídeo,
  pré-treinadas no Kinetics-400).
- **Entrada:** RGB (apenas imagem) ou **two-stream** (RGB + fluxo óptico de Farnebäck,
  com fusão tardia), para enfatizar o movimento.

## Pré-processamento
- 64 frames por clipe, redimensionados para 112×112.
- `full_frame` (frame inteiro). Opcionalmente, recorte da região dos atletas via YOLO.

## Treinamento (ajuste fino em 2 fases)
1. **Cabeça** (5 épocas): backbone congelado, `lr = 1e-4`.
2. **Ajuste fino** (40 épocas): backbone descongelado, `lr = 1e-5`, camadas iniciais
   mantidas congeladas.
- Regularização/otimização: `weight_decay = 0,01`, `dropout = 0,4`,
  `label_smoothing = 0,1`, escalonador cosseno com 2 épocas de warmup, precisão mista
  automática (AMP), acumulação de gradientes (4 passos), pesos de classe na perda.
- Semente: 42.

## Avaliação — validação cruzada por fonte
- **StratifiedGroupKFold** com 7 folds, estratificado por classe e agrupado por
  `source_id`: nenhuma fonte cruza treino/validação/teste (evita vazamento de
  near-duplicates).
- Métrica principal: **F1 macro**, reportado como **média ± desvio** entre os 7 folds.

## Resultado
| Modelo | Entrada | F1 macro (VC por fonte) |
|---|---|---|
| R(2+1)D-18 | RGB | 0,55 ± 0,09 |
| R(2+1)D-18 | RGB + fluxo | 0,65 ± 0,04 |
| **MC3-18** | **RGB + fluxo** | **0,68 ± 0,03** |

Comandos exatos em `scripts/02_treinar_cv_mc3.*`; hiperparâmetros por fold em
`results/cv_mc3_v2_40ep/foldN/config.json`.
