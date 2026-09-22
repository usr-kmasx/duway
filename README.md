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

**MCP (aba "outros" → config de mcp)**

- Tela com a lista dos servidores: nome/comando editáveis na hora, ativar/desativar, remover, **+ adicionar**, e editor direto do `~/.config/duway/mcp.json` (escrita atômica + backup `.bak`).
- **Padrão do app**: o server traz **`web-search` + `mcp-server-shell`** (shell de terminal p/ o modelo) — **todo boot confere** o `mcp.json` e volta a colocar o que faltar; quem já existe não é mexido (apagou um, ele volta no próximo boot; desligou um, continua desligado).
- **Instalar com um link** (barra de cima): cola `github.com/x/y`, `git+URL`, nome do PyPI ou `@pacote npm` e clica **instalar** (Enter também) — ele **baixa, sobe e confere** (`initialize` + `tools/list`) e **só grava no `mcp.json` se funcionou**; se falhar, mostra o log e não salva nada. Prefixos `npx:`/`uvx:` forçam o runtime, e dá pra colar a linha inteira (`npx -y pacote`).
- **npm/npx sem node no sistema**: na 1ª instalação npm ele baixa um **node LTS portátil** pra `duway/node/` (45M baixados, ~210M no disco, gitignored) e o cache do npm fica em `cache/npm/` — nada em `~/.npm`.
- **Erro com causa explicada em português**: repo **Go** → avisa que por link só rodam Python/Node (e que ele não publica binário); repo que **baixa mas não fala MCP** → avisa que pode ser um **cliente/CLI em vez de servidor**; repo **JS** (`package.json`) → tenta **npx automaticamente** e mostra `· o repo não é Python (é Node) — instalei como npm/npx`; linha de docs com `--with`/`--from` **sem o comando** → pede pra colar só o link; e se a conexão cair no meio, a tela **re-sincroniza a lista** e diz se a instalação completou no servidor.
- **Aplicar grava o arquivo e já mata/sobe os processos na hora** — diferente do config do llama, aqui não espera reinício nenhum.
- Tudo contido na pasta do projeto: `.venv/`, `cache/` (uv, pip, playwright, chromedriver, npm) e `TMPDIR` — nada é instalado no sistema.
- JSON-RPC por stdio; rotas `GET|POST /api/mcp`, `POST /api/mcp/instalar`, `GET /api/tools` e `POST /api/tool` prontas pro chat usar (o loop de tool_calls fica por sua conta / do modelo).
- O chat **envia as tools pro modelo**: ele pede a ferramenta → o site executa via `/api/tool` → devolve `role:"tool"` → resposta final (máx. 5 rodadas; balão tracejado `ferramenta · …` mostra args e resultado; só a resposta final entra no histórico).
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
