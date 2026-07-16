# detector_quedas — executável standalone

Empacotamento do detector de quedas (mesma lógica de `run_fall_batch_headless.py`,
sem alteração na detecção) como executável, para reprodutibilidade do TCC.

## Como funciona

O executável (`detector_quedas`) **não** embute `torch`/`ultralytics` — essas
duas dependências são pesadas (centenas de MB a poucos GB com CUDA) e
embuti-las deixaria o executável enorme e frágil entre máquinas/SOs
diferentes. Em vez disso:

1. Você instala `ultralytics` (que já traz `torch` como dependência) no seu
   próprio Python:
   ```
   pip install ultralytics
   ```
2. O executável faz uma checagem rápida no início. Se a dependência não
   estiver instalada, mostra um aviso claro com o comando exato pra rodar,
   e encerra sem travar/sem erro confuso.
3. Se estiver tudo certo, o executável delega o processamento pesado (YOLO
   pose + a lógica de detecção) para o **Python do sistema** via
   subprocesso — não para um interpretador Python empacotado dentro do
   executável. Isso evita problemas de biblioteca padrão incompleta que
   acontecem ao tentar importar bibliotecas pesadas dentro de um
   interpretador "congelado" (PyInstaller).

## Requisitos

- `pip install ultralytics` (traz `torch` junto) no Python que estiver no
  `PATH` como `python3` (ou informe outro com `--python /caminho/pra/python3`).
- `opencv-python` e `numpy` (também instalados automaticamente como
  dependência do restante do pipeline).
- **Conexão com a internet na primeira execução**: o peso `yolo11n-pose.pt`
  (modelo de pose oficial da Ultralytics, não treinado por este projeto) não
  vem junto nesta pasta — o próprio `ultralytics` baixa e faz cache dele
  automaticamente na primeira vez que rodar (arquivo oficial deles, direto
  do repositório `github.com/ultralytics/assets`). Isso evita redistribuir
  um artefato de terceiros e mantém a licença do que a gente entrega mais
  simples: só o código e o modelo próprio do projeto (`tatame_guard.pt`).

## Conteúdo desta pasta (mantenha tudo junto)

```
detector_quedas          <- o executável em si
worker_deteccao.py       <- roda no Python do sistema (não mexer)
modulos/                 <- lógica de detecção (fall-detector)
```

`yolo11n-pose.pt` (modelo oficial da Ultralytics, licença AGPL-3.0) **não**
está nesta pasta — é baixado automaticamente pelo `ultralytics` na primeira
execução e fica em cache no diretório onde o comando for rodado.

## Observação sobre o `tatame_guard`

Esta versão **não** inclui o `tatame_guard` (validação de que a queda ocorreu
dentro da área do tatame). Isso simplifica o pacote distribuído, mas muda o
comportamento: toda queda detectada pela pose é reportada, mesmo que tenha
ocorrido fora do tatame (plateia, árbitro, etc.) — sem essa filtragem extra,
o risco de falso positivo é maior.

## Uso

```bash
./detector_quedas \
  --list candidatos.json \
  --videos-base /caminho/pros/videos \
  --output relatorio_quedas.json
```

`candidatos.json`: lista no formato `[{"id": "...", "arquivo": "video1.mp4"}, ...]`,
onde `arquivo` é relativo a `--videos-base`.

Saída (`relatorio_quedas.json`): lista com os timestamps (mm:ss) de cada
queda detectada por vídeo — mesmo formato de `run_fall_batch_headless.py`.

## Testado em

- Linux (Ubuntu/Pop!_OS), Python 3.10, ultralytics instalado via pip.
- Verificado: (1) detecção real produz resultado idêntico ao script
  original (mesmo vídeo, mesmas 8 quedas); (2) aviso de dependência
  faltando aparece corretamente quando testado contra um venv sem
  ultralytics/torch instalados.
- **Não testado** em Windows/macOS — o mecanismo de achar o "Python do
  sistema" via `python3`/`python` no PATH deve funcionar, mas não foi
  verificado nessas plataformas.
