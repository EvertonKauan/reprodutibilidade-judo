# analysis_data — dados transformados pelos scripts

## `clips/` (o dataset de treino — 594 clipes)
Os **594 clipes** de ~4 segundos, um por golpe, com a **classe no nome do arquivo**:
```
ashi_waza_2M0AufUQqrY_luta03_sub02.mp4     -> fonte: 2M0AufUQqrY
sutemi_waza_3CMuw3ljBlQ_luta07_sub01.mp4   -> fonte: 3CMuw3ljBlQ
te_waza_JTTP_kAAX7k_luta03_sub03.mp4       -> fonte: JTTP_kAAX7k
```

Estes clipes **não** são versionados no repositório (direitos autorais dos vídeos-fonte),
mas são **regenerados automaticamente** a partir do manifesto:

```bash
python scripts/01_baixar_videos.py     # baixa os 13 vídeos-fonte (URLs do manifesto)
python scripts/05_gerar_clipes.py      # recorta os 594 clipes nos intervalos do manifesto
```

- A **classe** é inferida do nome (`sutemi_waza`, `ashi_waza`, `te_waza`).
- O **`source_id`** (vídeo-fonte de origem) sai do nome, permitindo o split **por fonte**
  no treino (`--group_split`), que evita vazamento de near-duplicates (clipes da mesma luta
  nunca caem em treino e teste ao mesmo tempo).

Aponte o treino para essa pasta:
```bash
bash scripts/02_treinar_cv_mc3.sh data/analysis_data/clips
```
