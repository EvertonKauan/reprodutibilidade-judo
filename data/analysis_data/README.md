# analysis_data — dados transformados pelos scripts

## `clips/` (subconjunto público — 594 clipes)
Os **594 clipes** (subconjunto público) de ~4 segundos, um por golpe, **recortados dos 22 vídeos-fonte** (ver
`../input_data/`) pelo extrator de highlights e **rotulados pela classe no nome do arquivo**:
```
ashi_waza_2M0AufUQqrY_luta03_sub02.mp4     -> fonte: 2M0AufUQqrY.mp4
sutemi_waza_3CMuw3ljBlQ_luta07_sub01.mp4   -> fonte: 3CMuw3ljBlQ.mp4
te_waza_JTTP_kAAX7k_luta03_sub03.mp4       -> fonte: JTTP_kAAX7k.mp4
```
- A **classe** é inferida do nome (`sutemi_waza`, `ashi_waza`, `te_waza`).
- O **`source_id`** (o vídeo-fonte de origem) sai do nome, no formato
  `<classe>_<idFonte>_luta<NN>_sub<NN>`. É o que liga cada clipe a uma das 22 fontes e
  permite o split **por fonte** no treino (`--group_split`), evitando vazamento de
  near-duplicates (clipes da mesma luta nunca caem em treino e teste ao mesmo tempo).

Obtenha os clipes de uma destas formas:
1. **Sob solicitação** aos autores (direitos autorais dos vídeos-fonte).
2. **Extrator** — rode o detector de highlights sobre os vídeos completos
   (`data/input_data/videos_fonte/`) para gerar os trechos.

Coloque os clipes em `analysis_data/clips/` e aponte o treino para essa pasta:
```bash
bash scripts/02_treinar_cv_mc3.sh data/analysis_data/clips
```

> O split train/val/test **não** é fixado por arquivo: é gerado no treino por
> `StratifiedGroupKFold` por fonte (seed 42). Assim o pacote funciona com qualquer
> subconjunto público de clipes — o resultado será **próximo** do original, não idêntico.
