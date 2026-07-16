# Classificação de sub-técnicas de judô em vídeo — MC3 two-stream

Material **reproduzível** (código, dados e modelo) do experimento de classificação de
golpes de judô em vídeo. Modelo temporal **MC3-18** com entrada **two-stream
(RGB + fluxo óptico)**, avaliado com **validação cruzada por fonte** (StratifiedGroupKFold,
sem vazamento de contexto).

## Resultado principal

| | Macro F1 (validação cruzada, 7 folds) |
|---|---|
| **MC3-18 two-stream — binário sutemi vs tachi** | **0,68 ± 0,03** (min 0,64 / máx 0,71) |

O número reportado é a **média da validação cruzada**, não o melhor fold (evita
cherry-picking). Detalhamento por fold em [`results/cv_summary.md`](results/cv_summary.md)
(gerado por `scripts/04_resumo_cv.py`).

> **Subconjunto público.** O experimento reportado usou o **conjunto completo** (~1012
> clipes de várias fontes, descrito no artigo), em parte **privado**. Este repositório
> disponibiliza apenas um **subconjunto público** — **594 clipes** de **22 vídeos-fonte**
> do YouTube — para demonstrar e verificar o pipeline. Treinar somente com esse subconjunto
> **não reproduz** o `0,68 ± 0,03`; os relatórios do experimento completo estão em
> `results/cv_mc3_v2_40ep/` como referência.

> **Por que validação cruzada por fonte?** Muitos clipes vêm da mesma luta/fonte
> (near-duplicates). Num split por vídeo, clipes da mesma luta caem em treino **e** teste,
> e o modelo memoriza a cena (métrica inflada). O `StratifiedGroupKFold` por fonte garante
> que **nenhuma fonte cruza** as partições → número honesto. Esse é o principal cuidado
> metodológico do trabalho.

## Estrutura do repositório

```
judo-subtecnicas-mc3/
├── src/
│   └── main.py                     # Código de treino/avaliação (modelos temporais)
├── data/
│   ├── input_data/                 # Originais e intocáveis
│   │   ├── fontes_videos.csv        #   22 vídeos-fonte públicos (YouTube)
│   │   └── videos_fonte/            #   vídeos completos (baixados via URL)
│   └── analysis_data/
│       └── clips/                   #   clipes de ~4s rotulados (sob solicitação / extrator)
├── scripts/
│   ├── 01_baixar_videos.py          # baixa os vídeos-fonte (yt-dlp)
│   ├── 02_treinar_cv_mc3.sh/.ps1    # treino com validação cruzada (7 folds)
│   ├── 03_... calibracao_mc3_v2.py  # calibração de limiar (opcional)
│   └── 04_resumo_cv.py              # média ± desvio da CV (número do artigo)
├── results/
│   ├── models/                      # peso do melhor fold (.pt)
│   ├── cv_mc3_v2_40ep/foldN/        # métricas por fold (report, matriz, curvas)
│   ├── calibracao/                  # saídas da calibração
│   └── cv_summary.md                # tabela-resumo da CV
├── docs/                            # metodologia e notas de laboratório
├── requirements.txt / environment.yml
├── LICENSE  ·  CITATION.cff  ·  README.md
```

## 1) Ambiente

Requer **GPU NVIDIA (CUDA)** para treinar (o modelo é temporal 3D). Python 3.10.

```bash
# opção A: pip
pip install -r requirements.txt
#   torch com CUDA: veja a nota no topo do requirements.txt

# opção B: conda
conda env create -f environment.yml
conda activate judo-mc3
```

## 2) Dados

Este repositório compartilha um **subconjunto público** do dataset do estudo (o conjunto completo, maior e em parte privado, é descrito no artigo). Os **vídeos-fonte não são redistribuídos** (direitos autorais). O
arquivo [`data/input_data/fontes_videos.csv`](data/input_data/fontes_videos.csv) traz as
**URLs** dos **22 vídeos-fonte** (públicos, YouTube), de onde os **594 clipes** rotulados
foram recortados:

| Classe (binária) | Composição | Clipes |
|---|---|---|
| Sutemi Waza | sutemi_waza | 175 |
| Tachi Waza | ashi_waza (245) + te_waza (174) | 419 |
| **Total** | — | **594** |

A lista completa das **22 fontes** está em
[`data/input_data/fontes_videos.csv`](data/input_data/fontes_videos.csv).

Os **clipes** rotulados (dataset de treino) vão em `data/analysis_data/clips/`. Por
restrições de direitos autorais, **não** são publicados neste repositório e podem ser
**solicitados aos autores**. Cada clipe é rastreável à sua fonte pelo nome:
`<classe>_<idFonte>_luta<NN>_sub<NN>.mp4` — ex.: `te_waza_JTTP_kAAX7k_luta03_sub03.mp4` veio
de `JTTP_kAAX7k`.

Para (re)obter os dados:
```bash
# (a) baixar os 22 vídeos-fonte direto do YouTube (720p)
python scripts/01_baixar_videos.py

# (b) gerar os clipes a partir dos vídeos-fonte com o extrator de highlights (~4 s por golpe)
```

A classe está no nome do arquivo; o split por fonte é gerado no treino (`--group_split`).
Detalhes em [`data/README.md`](data/README.md).

## 3) Treino (validação cruzada, 7 folds)

```bash
# Linux (GPU):
bash scripts/02_treinar_cv_mc3.sh data/analysis_data/clips

# Windows (GPU):
powershell -ExecutionPolicy Bypass -File scripts\02_treinar_cv_mc3.ps1 data\analysis_data\clips
```

Cada fold reproduz exatamente os hiperparâmetros de
`results/cv_mc3_v2_40ep/foldN/config.json` (mc3_18, binário sutemi vs tachi, 64 frames,
112 px, two-stream RGB+flow, 40 épocas de fine-tuning, seed 42, split por fonte).

## 4) Número final da CV

```bash
python scripts/04_resumo_cv.py      # imprime e salva results/cv_summary.md
```

## Modelo treinado

`results/models/mc3_18_fusion_fold5_best_macroF1_0.720.pt` — pesos do **melhor fold**
(fold 5, macro F1 0,715). Os pesos dos demais folds não são redistribuídos por tamanho; os **relatórios** de todos os 7 folds estão em
`results/cv_mc3_v2_40ep/`.

## Documentação adicional

- [`docs/metodologia.md`](docs/metodologia.md) — resumo reprodutível: dataset, modelos,
  entrada two-stream, pré-processamento, treino em 2 fases e validação cruzada por fonte.

## Citação

Veja [`CITATION.cff`](CITATION.cff).

## Licença

Código sob [MIT](LICENSE). Os vídeos-fonte pertencem aos canais originais do YouTube e são
usados apenas para pesquisa acadêmica (ver nota no `LICENSE`).
