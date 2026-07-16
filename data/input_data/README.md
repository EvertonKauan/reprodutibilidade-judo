# input_data — dados originais e intocáveis

## `fontes_videos.csv`
Os **22 vídeos-fonte** (públicos, YouTube) a partir dos quais os **594 clipes** do subconjunto público
foram recortados. Colunas: `id`, `canal`, `url`, `arquivo`, `clipes_derivados`.

| Vídeo-fonte (`arquivo`) | Canal | Clipes derivados |
|---|---|---|
| Sutemi Waza | `sutemi_waza` | 175 |
| Tachi Waza | `ashi_waza` + `te_waza` | 419 |
| **Total** | | **594** |

## `videos_fonte/` (os 22 vídeos completos)
São a **matéria-prima**: cada um dá origem a vários clipes. Obtenha de uma destas formas:
1. **URLs** — as URLs dos 22 vídeos estão em `fontes_videos.csv`.
2. **Download** — baixe direto do YouTube:
   ```bash
   python scripts/01_baixar_videos.py        # 720p (padrao)
   ```
   Salvos como `videos_fonte/<id>.mp4`.

Do vídeo completo, o **extrator de highlights** recorta os trechos de ~4 s (um por golpe),
que viram os clipes rotulados em `../analysis_data/clips/`.

> **Rastreabilidade fonte → clipe:** o nome de cada clipe embute o `id` da fonte
> (`<classe>_<idFonte>_luta<NN>_sub<NN>.mp4`). Assim, todo clipe é ligado ao seu vídeo-fonte
> — e é isso que permite o split **por fonte** no treino (nenhuma fonte cruza train/teste).

> **Direitos:** os vídeos pertencem aos canais originais do YouTube (uso acadêmico/pesquisa).
