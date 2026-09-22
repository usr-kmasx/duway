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

### Celular (mesma rede)

```sh
python3 server.py --host 0.0.0.0
# o boot mostra a url, ex.: http://192.168.100.9:8787
```

O chat, o llama (via proxy do site) e os MCPs funcionam no celular sem expor o llama-server. **Sem senha**: só em rede confiável — qualquer um no Wi-Fi abre o site e chama a API. Sem a flag, é só-PC (`127.0.0.1`). Se não abrir: `sudo ufw allow 8787/tcp`, mesmo Wi-Fi (fora dados móveis/VPN/rede de visitas).

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

**MCP (aba "outros" → config de mcp)**

- Tela com a lista dos servidores: nome/comando editáveis na hora, ativar/desativar, remover, **+ adicionar**, e editor direto do `~/.config/duway/mcp.json` (escrita atômica + backup `.bak`). O caminho do arquivo aparece no topo da tela.
- **Padrão do app**: o server traz **`web-search` + `mcp-server-shell`** (shell de terminal p/ o modelo) — **todo boot confere** o `mcp.json` e volta a colocar o que faltar (completando o `uso` do shell se sumiu); quem já existe não é mexido (apagou um, ele volta no próximo boot; desligou um, continua desligado).
- **Sem instalação pelo site**: não há mais campo de link nem rota de instalar — mcp novo entra **baixando você mesmo e editando o `mcp.json`** (na tela ou direto em `~/.config/duway/mcp.json`); o **aplicar valida e já mata/sobe os processos na hora**.
- **Config de uso por mcp (`uso`)**: o shell nasce com `base` (diretório de trabalho do processo, editável na linha), `pedir_fora` e duas listas — **`permitidos`** (começo cadastrado passa sem pedir, nem fora da base: `ls -la` libera `ls -la ~/x`) e **`sempre_pedir`** (`rm`, `sudo`, `su`, `passwd` por padrão: começo do comando que **pede até dentro da base** — os padrões aparecem travados na janela, sem ✕; o que você adicionar sai). Fora da base, o resto **pede** (permitir 1x, liberar sessão ou recusar — **nada é bloqueado**). Janela própria no botão **permitidos…** da linha do shell; vale pra qualquer mcp via JSON.
- **Aplicar grava o arquivo e já mata/sobe os processos na hora** — diferente do config do llama, aqui não espera reinício nenhum.
- Tudo contido na pasta do projeto: `.venv/`, `cache/` (uv, pip, playwright, chromedriver) e `TMPDIR` — nada é instalado no sistema.
- JSON-RPC por stdio; rotas `GET|POST /api/mcp`, `GET /api/tools` e `POST /api/tool` prontas pro chat usar (o loop de tool_calls fica por sua conta / do modelo).
- O chat **envia as tools pro modelo**: ele pede a ferramenta → o site executa via `/api/tool` → devolve `role:"tool"` → resposta final (máx. 5 rodadas; balão tracejado `ferramenta · …` mostra args e resultado, e **clicar nele abre a janela `chamada`** com ferramenta, servidor, estado, args JSON completo e resultado completo; só a resposta final entra no histórico).
- `bin/google-chrome` é um **shim pro chromium** (o webdriver-manager procura `google-chrome`; sem ele ele baixa o driver errado e a busca vem vazia) e injeta **`--headless`** — sem isso o MCP abre uma janela na tela, e fechar essa janela mata a sessão selenium. Esse PATH vale só pros processos MCP.
- **Supervisor**: se a sessão selenium morrer (o MCP tem o bug de nunca recriar o driver), o `server.py` reinicia o processo MCP (filho nosso, nunca o llama) e **repete a chamada uma vez** — na UI aparece `· (mcp reiniciado)`.
- O `GET /api/mcp` traz **`log`** — as últimas linhas do stderr de cada MCP, pra diagnosticar quando der erro.

**Visual**

- Preto por padrão; ◐ inverte o tema (salvo + deep links `#dark` / `#light` / `#config`).
- Textarea sem scrollbar em repouso, cresce até 160px.

## Estrutura

| arquivo        | papel                                                        |
| -------------- | ------------------------------------------------------------ |
| `index.html`   | interface inteira (JS vanilla, IndexedDB, tema, menu, histórico) |
| `server.py`    | backend stdlib: estático + `GET /api/flags` + `GET\|POST /api/config` + 5 rotas MCP (processos filhos) |
| `history.png`  | origem do ícone de histórico (vira data-URI)                  |

## Notas

- O `config.ini` é lido pelo llama-server **só no startup** (precedência: `config.ini` < env < CLI).
- MCP é isolado na **instalação** (venv e cache do projeto), não na **ação**: um servidor MCP roda com seus privilégios — só adicione o que você confiar.
- Porta do site (`server.py`) e porta do llama são coisas diferentes: a do site é a do navegador, a do llama vem do config/menu.
