# duway

Chat minimalista preto/branco que fala com um `llama-server` local (llama.cpp).
Sem framework e sem dependência: um `index.html` e um `server.py` — só Python 3 stdlib e JS puro.

## Rodando

```sh
cd ~/Projetos/duway
python3 server.py
# abre http://127.0.0.1:8787
```

Pré-requisito: um llama-server rodando (porta configurável no menu ☰):

```sh
llama-server -m modelo.gguf --mmproj mmproj.gguf
```

## O que tem

**Conversa**

- Streaming com **tk/s** e **ctx %** ao vivo no topo; nome do modelo detectado sozinho (segue a troca do `.gguf` sem recarregar).
- Botão **novo**: abre chat novo e zera ctx/tk-s. O ctx é **por conversa** — reabrir uma conversa antiga restaura o ctx dela (o modelo recebe o histórico todo).
- **Pensamento por balão**: bolinha antes do nome do modelo — pisca enquanto ele pensa, fica fixa depois; clicar no balão abre a janela com o pensamento **daquela resposta** (✕, Esc ou clique fora fecha). Persiste no histórico.
- **Histórico** de conversas (IndexedDB): abrir, selecionar, apagar; imagens persistem no F5; retomada pós-refresh.

**Anexos**

- Arrastar (ou colar print) direto pro modelo: imagens, vídeo (extrai quadros) e texto.
- Clique na miniatura abre o lightbox.

**Config (menu ☰)**

- 254 flags do llama em 9 abas, com busca.
- Lê e grava `~/.config/llama.cpp/config.ini` (escrita atômica + backup `.bak`).
- **Aplicar só grava o arquivo** — o site nunca reinicia o llama-server sozinho. Como o config é lido no startup, **a mudança só vale depois que você reiniciar o llama-server manualmente**.

**Visual**

- Preto por padrão; ◐ inverte o tema (salvo + deep links `#dark` / `#light` / `#config`).
- Textarea sem scrollbar em repouso, cresce até 160px.

## Estrutura

| arquivo        | papel                                                        |
| -------------- | ------------------------------------------------------------ |
| `index.html`   | interface inteira (JS vanilla, IndexedDB, tema, menu, histórico) |
| `server.py`    | backend stdlib: estático + `GET /api/flags` + `GET\|POST /api/config` |
| `history.png`  | origem do ícone de histórico (vira data-URI)                  |

## Notas

- O `config.ini` é lido pelo llama-server **só no startup** (precedência: `config.ini` < env < CLI).
- Porta do site (`server.py`) e porta do llama são coisas diferentes: a do site é a do navegador, a do llama vem do config/menu.
