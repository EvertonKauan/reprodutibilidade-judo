# Classificação de sub-técnicas de judô em vídeo — MC3 two-stream

Material **reproduzível** (código, dados e modelo) do experimento de classificação de
golpes de judô em vídeo. Modelo temporal **MC3-18** com entrada **two-stream
(RGB + fluxo óptico)**, avaliado com **validação cruzada por fonte** (StratifiedGroupKFold,
sem vazamento de contexto).

## Resultado principal

| Modelo (two-stream, 7-fold CV por fonte) | Macro F1          |
| ---------------------------------------- | ----------------- |
| **MC3-18 — binário sutemi vs tachi**     | **0,698 ± 0,029** |
| R(2+1)D-18 (mesmo split/dataset)         | 0,590 ± 0,054     |

O número reportado é a **média da validação cruzada**, não o melhor fold (evita
cherry-picking). Detalhamento por fold em [`results/cv_summary.md`](results/cv_summary.md)
(gerado por `scripts/04_resumo_cv.py`).

> **Amostra pública (não reproduz 100%).** O experimento reportado usou o **conjunto
> completo** (1392 clipes: 527 sutemi + 859 tachi, descrito no artigo), em parte **privado**.
> **Nem todos os vídeos são públicos.** Este repositório disponibiliza apenas uma **amostra
> pública** — **594 clipes** de **13 vídeos-fonte** do YouTube — para demonstrar e verificar
> o pipeline. Treinar somente com essa amostra **não reproduz** o `0,698 ± 0,029`; os
> relatórios completos estão em `results/cv_mc3_v2_40ep/` como referência.

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
│   │   ├── fontes_videos.csv        #   MANIFESTO: 1 linha por clipe (id, url, momento_corte, classe, arquivo)
│   │   └── videos_fonte/            #   vídeos completos (baixados via URL)
│   └── analysis_data/
│       └── clips/                   #   clipes de ~4s (regeneráveis do manifesto via script 05)
├── scripts/
│   ├── 01_baixar_videos.py          # baixa os vídeos-fonte (yt-dlp)
│   ├── 05_gerar_clipes.py           # recorta os 594 clipes do manifesto (ffmpeg)
│   ├── 02_treinar_cv_mc3.sh/.ps1    # treino com validação cruzada (7 folds)
│   ├── 03_... calibracao_mc3_v2.py  # calibração de limiar (opcional)
│   └── 04_resumo_cv.py              # média ± desvio da CV (número do artigo)
├── results/
│   ├── models/                      # peso do melhor fold (.pt)
│   ├── cv_mc3_v2_40ep/foldN/        # métricas por fold (report, matriz, curvas)
│   ├── calibracao/                  # saídas da calibração
│   └── cv_summary.md                # tabela-resumo da CV
├── dist/                            # extrator de quedas standalone (Linux) — ver dist/README.md
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

Este repositório compartilha um **subconjunto público** do dataset do estudo (o conjunto
completo, maior e em parte privado, é descrito no artigo). Os **vídeos-fonte não são
redistribuídos** (direitos autorais dos canais), mas os **594 clipes são totalmente
regeneráveis** a partir do manifesto — **sem precisar entrar em contato com os autores**.

O manifesto [`data/input_data/fontes_videos.csv`](data/input_data/fontes_videos.csv) tem
**uma linha por clipe**, com o vídeo-fonte (`id`, `url`), o **intervalo de corte**
(`momento_corte`), a `classe` e o nome do `arquivo`. Os 594 clipes vêm de **13 vídeos-fonte
públicos** do YouTube.

| Classe (binária) | Composição                      | Clipes  |
| ---------------- | ------------------------------- | ------- |
| Sutemi Waza      | sutemi_waza                     | 175     |
| Tachi Waza       | ashi_waza (245) + te_waza (174) | 419     |
| **Total**        | —                               | **594** |

**Para reconstruir o dataset** (não precisa de contato):

```bash
# (a) baixar os 13 vídeos-fonte a partir das URLs do manifesto (720p)
python scripts/01_baixar_videos.py

# (b) recortar os 594 clipes nos intervalos exatos do manifesto (requer ffmpeg)
python scripts/05_gerar_clipes.py
```

Os clipes ficam em `data/analysis_data/clips/`, prontos para o treino. A classe está no
nome do arquivo; o split por fonte é gerado no treino (`--group_split`). Detalhes em
[`data/README.md`](data/README.md).

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

## Extrator de quedas (`dist/`)

A pasta [`dist/`](dist/) traz um **executável standalone** (`detector_quedas`) que detecta
as quedas em vídeos completos por pose (YOLO). É uma ferramenta **opcional** de apoio à
extração de clipes: você pode usá-la **ou** recortar manualmente. Detalhes em
[`dist/README.md`](dist/README.md).

- **Sobredetecta:** por ser baseado em pose, marca quedas a mais do que o necessário (ex.:
  atletas em pé, mas curvados). Os trechos incorretos precisam ser **revisados e descartados
  manualmente**.
- **Não embute** `torch`/`ultralytics` no binário; delega ao Python do sistema (você instala
  `ultralytics` com `pip`). O modelo de pose oficial da Ultralytics é baixado por ela e
  **não** é redistribuído aqui — só o código próprio do projeto acompanha o pacote.
- Compilado e **testado em Linux** (ELF); não verificado em Windows/macOS.

## Modelo treinado

`results/models/mc3_18_fusion_fold5_best_macroF1_0.720.pt` — pesos do **melhor fold**
(fold 5, macro F1 0,715). Os pesos dos demais folds não são redistribuídos por tamanho; os **relatórios** de todos os 7 folds estão em
`results/cv_mc3_v2_40ep/`.

## Documentação adicional

- [`docs/metodologia.md`](docs/metodologia.md) — resumo reprodutível: dataset, modelos,
  entrada two-stream, pré-processamento, treino em 2 fases e validação cruzada por fonte.

## Citação

Veja [`CITATION.cff`](CITATION.cff).
