# Parâmetros do fluxo óptico (Farnebäck) e formato do cache

## Chamada `cv2.calcOpticalFlowFarneback`

Chamada em `main.py` (dentro de `compute_flow_uv`): `cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)`

(assinatura confirmada na docstring do OpenCV instalado nesta máquina, versão 4.11.0)

| Posição | Parâmetro | Valor usado | Significado |
|---|---|---|---|
| 1 | `prev` | `a` | Frame grayscale no instante t |
| 2 | `next` | `b` | Frame grayscale no instante t+1 |
| 3 | `flow` | `None` | Saída alocada internamente pelo OpenCV (não reaproveita buffer) |
| 4 | `pyr_scale` | `0.5` | Fator de escala entre níveis da pirâmide; 0.5 = pirâmide clássica, cada nível com metade do tamanho do anterior |
| 5 | `levels` | `3` | Número de níveis da pirâmide, incluindo a imagem original (permite captar deslocamentos maiores que o `winsize` em um único nível) |
| 6 | `winsize` | `15` | Tamanho da janela de média usada na estimativa; maior = mais robusto a ruído e a movimento rápido, mas campo de fluxo mais borrado |
| 7 | `iterations` | `3` | Número de iterações do algoritmo em cada nível da pirâmide |
| 8 | `poly_n` | `5` | Tamanho da vizinhança de pixels usada na expansão polinomial local |
| 9 | `poly_sigma` | `1.2` | Desvio-padrão do Gaussiano usado para suavizar as derivadas da expansão polinomial (valor recomendado pela doc do OpenCV para `poly_n=5` é 1.1; aqui é 1.2) |
| 10 | `flags` | `0` | Nenhuma flag ativada — usa filtro de caixa (não Gaussiano) e não usa fluxo inicial |

Referência do algoritmo: Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA 2003.

## 1,53 MiB por clipe — derivação exata

Não está escrito em lugar nenhum do repo como número — é derivado do formato do cache e confirmado contra um arquivo real:

- Tensor cacheado por clipe: `[T, H, W, 2]` em `uint8` → `_quantize_flow()` (`main.py`) retorna exatamente isso (2 canais, u e v; a magnitude é derivada em `flow_uv_to_tensor()` no load, nunca persistida).
- Com `T=64`, `H=W=112`: payload bruto = 64 × 112 × 112 × 2 × 1 byte = 1.605.632 bytes.
- Formato de arquivo: `np.save()` (não `savez_compressed`) → grava um `.npy` v1.0, que soma um cabeçalho (magic string + versão + dict de shape/dtype, com padding para alinhamento de 64 bytes) — nesse caso 128 bytes.
- Total esperado: 1.605.632 + 128 = 1.605.760 bytes.

Confirmado contra um arquivo real do cache de produção na VM:

```
$ ls -la flow_cache_pro_nf64_sz112/te_waza_lX7brd9T1-s_luta37_sub03__nf64_sz112_ycrop_static.npy
-rw-r--r-- 1 vpcuser vpcuser 1605760 Jul 11 16:58 ...npy
```

1.605.760 bytes = 1,5313 MiB ≈ 1,53 MiB, batendo exato (0 bytes de diferença) com o cálculo.
