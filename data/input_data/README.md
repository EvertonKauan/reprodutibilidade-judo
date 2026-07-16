# input_data — dados originais e intocáveis

## `fontes_videos.csv` (o manifesto)
Manifesto do subconjunto público: **uma linha por clipe** (594 no total), permitindo
regenerar todo o dataset a partir dos vídeos públicos. Colunas:

| Coluna | Descrição |
|---|---|
| `id` | ID do vídeo-fonte no YouTube |
| `canal` | Canal de origem (CBJ TV, Ochiru, etc.) |
| `url` | URL do vídeo-fonte |
| `momento_corte` | Intervalo do clipe no vídeo-fonte (`H:MM:SS.ss -> H:MM:SS.ss`) |
| `classe` | `sutemi_waza`, `ashi_waza` ou `te_waza` |
| `arquivo` | Nome do clipe gerado |

Os 594 clipes vêm de **13 vídeos-fonte públicos** distintos (dos canais CBJ TV, Ochiru,
Judo Highlights e Judo Spirit). Composição binária: **Sutemi 175 / Tachi 419**
(ashi 245 + te 174).

## `videos_fonte/` (os vídeos completos)
A **matéria-prima**. Não são redistribuídos (direitos autorais); baixe a partir das URLs
do manifesto:
```bash
python scripts/01_baixar_videos.py        # 720p (baixa cada fonte 1x)
```
Salvos como `videos_fonte/<id>.mp4`.

## Como os clipes são regenerados
Com os vídeos-fonte baixados, `scripts/05_gerar_clipes.py` recorta cada clipe no intervalo
exato do `momento_corte`, gravando em `../analysis_data/clips/`. Assim o dataset é
reconstruído **sem contato com os autores**.

> **Rastreabilidade:** o nome do clipe embute o `id` da fonte
> (`<classe>_<idFonte>_luta<NN>_sub<NN>.mp4`), o que também sustenta o split **por fonte**
> no treino (nenhuma fonte cruza train/teste).
