# Dados

Este repositório traz um **subconjunto público** (594 clipes de 22 vídeos-fonte) do dataset do estudo — o conjunto completo é maior e em parte privado (ver artigo). Organiza-se assim:

1. **22 vídeos-fonte completos** (públicos, do YouTube) — a matéria-prima, recortada pelo
   **extrator de highlights** para gerar os clipes.
2. **594 clipes já cortados e rotulados** — prontos para treinar o modelo; cada um rastreável
   a uma das 22 fontes pelo nome do arquivo.

Segue o padrão de compêndio reproduzível:

- **`input_data/`** — dados **originais e intocáveis**: a lista de fontes
  (`fontes_videos.csv`) e, após baixar/obter, os **vídeos completos** em `videos_fonte/`.
- **`analysis_data/`** — dados **transformados**: os **clipes** de ~4 s recortados dos
  vídeos completos (é a entrada do treino). Obtidos via URLs dos vídeos-fonte ou gerados pelo extrator.

> Nada em `input_data/` é editado. As transformações geram saída em `analysis_data/`
> ou em `results/`.

## Fluxo de dados

```
fontes_videos.csv ──(scripts/01_baixar_videos.py)──> input_data/videos_fonte/*.mp4
                                                              │
                          (extrator de highlights recorta os trechos dos golpes)
                                                              ▼
                                              analysis_data/clips/*.mp4   (rotulados)
                                                              │
                              (main.py: StratifiedGroupKFold por fonte, no treino)
                                                              ▼
                                                     treino / validação cruzada
```

## Reprodução aproximada (não idêntica)

Só os **vídeos públicos** são disponibilizados. Parte dos dados usados no experimento
original era privada e **não** é redistribuída. Além disso, o split é **regenerado**
automaticamente (`--group_split`) sobre os clipes disponíveis. Portanto, ao rodar este
pacote você obtém um resultado **próximo**, não exatamente o `0,68 ± 0,03` do artigo.
Os relatórios do experimento original ficam em `../results/cv_mc3_v2_40ep/` como referência.
