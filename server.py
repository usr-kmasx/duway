#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duway — chat minimalista para llama.cpp (backend local)

Serve o index.html e expõe 3 rotas:

  GET  /api/flags   -> roda `llama-server --help` e devolve TODAS as flags parseadas
  GET  /api/config  -> le o config.ini (aceita ?path=...; cria o padrao so se for o
                       caminho padrao ~/.config/llama.cpp/config.ini)
  POST /api/config  -> grava no caminho que o site mandou (com backup .bak)

  o caminho e escolhido pelo site (campo em cima do menu). O llama-server nao e
  afetado por isso: ele continua lendo o config do caminho dele por conta propria.

uso:
  python3 server.py            # sobe em http://127.0.0.1:8787
  python3 server.py --porta 9000
  LLAMA_BIN=/caminho/llama-server python3 server.py
  python3 server.py --dump-flags   # só imprime o JSON das flags (debug)
"""

import json
import os
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------- configuração

HOME = Path.home()
ROOT = Path(__file__).resolve().parent

LLAMA_BIN = os.environ.get("LLAMA_BIN", "")
CONFIG_PATH = Path(
    os.environ.get("XDG_CONFIG_HOME") or (HOME / ".config")
) / "llama.cpp" / "config.ini"
LISTEN_PORT = int(os.environ.get("DUWAY_PORT") or 8787)

DEFAULT_CONFIG = """\
# ~/.config/llama.cpp/config.ini
# config padrao do llama.cpp - lida automaticamente na inicializacao
# precedencia: config.ini  <  variaveis de ambiente  <  argumentos de CLI
# somente as secoes [*] e a padrao sao usadas; secoes nomeadas sao ignoradas
# editavel pelo menu (hamburguer) do duway - botao APLICAR
# nao coloque `model` aqui: passe -m na linha de comando

[*]
# --- servidor ---
host              = 127.0.0.1
port              = 8080
timeout           = 3600
threads-http      = -1
sse-ping-interval = 3600
cors-origins      = *
jinja             = 1

# --- contexto / cache (valores ja usados no seu llama-server) ---
ctx-size        = 64000
ubatch-size     = 512
flash-attn      = on
cache-type-k    = q4_0
cache-type-v    = q4_0
spec-draft-n-max = 3

# --- amostragem padrao ---
temp            = 0.80
top-k           = 40
top-p           = 0.95
min-p           = 0.05
repeat-penalty  = 1.00
"""

# abas do menu (ordem de exibicao)
TABS = [
    ("modelo", "Modelo & template"),
    ("contexto", "Contexto & KV"),
    ("gpu", "GPU & dispositivo"),
    ("cpu", "CPU & threads"),
    ("amostragem", "Amostragem"),
    ("spec", "Speculative decoding"),
    ("servidor", "Servidor & API"),
    ("log", "Log & diversos"),
    ("geral", "Outros"),
]

# flags que so executam algo e nao devem ir para o config.ini
ACTIONS = {
    "help", "usage", "version", "completion-bash", "cache-list", "list-devices",
}

SERVIDOR_KEYS = {
    "host", "port", "reuse-port", "path", "cors-origins", "cors-methods",
    "cors-headers", "cors-credentials", "api-prefix", "ui-config",
    "ui-config-file", "ui-mcp-proxy", "ui", "tools", "tools-runtime",
    "mcp-servers-config", "mcp-servers-json", "agent", "embedding",
    "embeddings", "rerank", "reranking", "api-key", "api-key-file",
    "ssl-key-file", "ssl-cert-file", "timeout", "sse-ping-interval",
    "threads-http", "cache-prompt", "cache-reuse", "metrics", "props",
    "slots", "slot-save-path", "slot-prompt-similarity", "models-dir",
    "models-preset", "models-max", "models-autoload", "media-path",
    "sleep-idle-seconds", "log-prompts-dir", "tags",
}

GPU_KEYS = (
    "gpu-layers", "device", "split-mode", "tensor-split", "main-gpu", "numa",
    "override-tensor", "op-offload", "repack", "cpu-moe", "n-cpu-moe",
    "n-cpu-ffn", "no-host", "fit", "flash-attn",
)

CPU_KEYS = ("threads", "cpu-mask", "cpu-range", "cpu-strict", "prio", "poll")

CONTEXTO_KEYS = (
    "ctx-size", "keep", "rope-", "yarn-", "swa-full", "parallel",
    "cont-batching", "context-shift", "kv-", "cache-type", "defrag-thold",
    "load-mode", "lazy-mode", "cache-ram", "ctx-checkpoints", "swa-checkpoints",
    "checkpoint-min-step", "cache-idle-slots", "predict", "batch-size",
    "ubatch-size",
)

MODELO_KEYS = (
    "model", "hf-", "docker-repo", "lora", "control-vector", "mmproj",
    "pooling", "embd-", "chat-template", "jinja", "skip-chat-parsing",
    "prefill-assistant", "spm-infill", "special", "warmup", "fim-", "gpt-oss",
    "vision-", "reasoning", "alias", "image-", "mtmd-", "video-",
)

LOG_KEYS = (
    "log-", "verbose", "verbosity", "offline", "perf", "escape",
    "check-tensors", "override-kv", "reverse-prompt", "lookup-cache",
)


# ---------------------------------------------------------------- parse --help

def find_llama_bin():
    candidatos = [
        LLAMA_BIN,
        str(HOME / "llama.cpp" / "build" / "bin" / "llama-server"),
        shutil.which("llama-server") or "",
    ]
    for c in candidatos:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return ""


def run_help(binary):
    import subprocess
    try:
        p = subprocess.run(
            [binary, "--help"],
            capture_output=True, text=True, timeout=30, errors="replace",
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("nao consegui rodar `%s --help`: %s" % (binary, e))
    out = p.stdout or ""
    if "-----" not in out:  # ajuda foi pro stderr
        out = p.stderr or ""
    if "-----" not in out:
        raise RuntimeError("saida de --help vazia")
    return out


def split_head(line):
    """'flag-flag flag N        descricao' -> ('flag-flag flag N', 'descricao')

    O llama alinha a descricao na coluna 40. Quando as flags nao cabem em 40
    colunas a descricao desce para a proxima linha (ai devolvemos desc='').
    """
    if len(line) > 40 and line[39] == " " and line[40] != " " and not line[40:].startswith("-"):
        head = line[:40].rstrip()
        desc = line[40:]
        if head:
            return head, desc
    return line.rstrip(), ""


def parse_flags(binary):
    text = run_help(binary)
    lines = text.splitlines()
    sec_re = re.compile(r"^-{3,}\s*(.+?)\s*-{3,}\s*$")

    entries = []
    section = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        m = sec_re.match(line)
        if m:
            section = m.group(1).strip()
            i += 1
            continue
        if line.startswith("-"):
            head, desc = split_head(line)
            j = i + 1
            extra = []
            while j < len(lines):
                nxt = lines[j]
                if nxt.startswith("-") or (sec_re.match(nxt) and sec_re.match(nxt).group(1)):
                    break
                if nxt.strip():
                    extra.append(nxt.strip())
                j += 1
            if desc:
                extra.insert(0, desc)
            entries.append((section, head, "\n".join(extra)))
            i = j
            continue
        i += 1

    flags = []
    sections = []
    for order, (sec, head, desc) in enumerate(entries):
        f = build_flag(sec, head, desc, order)
        if f:
            flags.append(f)
            if sec not in sections:
                sections.append(sec)
    return flags, sections


def build_flag(section, head, desc, order):
    # separa as flags do metavar: o metavar e o primeiro token que nao e flag
    parts = head.split()
    idx = next((n for n, t in enumerate(parts) if not t.startswith("-")), len(parts))
    longs = [t.strip(",") for t in parts[:idx] if t.startswith("--")]
    shorts = [t.strip(",") for t in parts[:idx] if t.startswith("-") and not t.startswith("--")]
    if not longs and not shorts:
        return None
    metavar = " ".join(parts[idx:]).strip().strip(",").strip()
    for opener, closer in (("[", "]"), ("{", "}"), ("<", ">")):
        if metavar.startswith(opener) and metavar.endswith(closer):
            metavar_inner = metavar[1:-1]
            break
    else:
        metavar_inner = metavar

    # tem par positivo E negativo?  (so --no-x = flag de presenca, liga e pronto)
    positivos = [l for l in longs if not l.startswith("--no-")]
    neg = bool(positivos) and any(l.startswith("--no-") for l in longs)
    key = (positivos[0] if positivos else (longs[0] if longs else shorts[0])).lstrip("-")
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.-]*$", key):
        return None

    flat = " ".join(desc.split())
    mdef = re.search(r"\(default:\s*(.*?)\)(?:\s|$)", flat)
    default = mdef.group(1).strip() if mdef else ""
    menv = re.search(r"\(env:\s*([A-Z0-9_]+)\)", flat)
    env = menv.group(1) if menv else ""

    # escolhas: [a|b|c]  {a,b,c}  <a|b>  e listas com marcador "- x:" na descricao
    choices = []
    if "|" in metavar_inner:
        choices = [c.strip() for c in metavar_inner.split("|") if c.strip()]
    elif metavar.startswith("{") and "," in metavar_inner:
        choices = [c.strip() for c in metavar_inner.split(",") if c.strip()]
    if not choices:
        choices = re.findall(r"^\s*-\s+([A-Za-z0-9_.+-]+)\s*:", desc, re.M)

    kind = "text"
    if not metavar:
        kind = "bool"
    elif choices:
        kind = "select"
    else:
        default_numeric = is_num(default.split(",")[0].split(" ")[0])
        metavar_num = bool(re.match(r"^(N\d*|SECONDS|PORT|INDEX|SEED)$", metavar))
        # "auto"/"all" sao valores validos de --gpu-layers e afins: deixa texto
        livre = ("auto" in desc.lower()) or ("'all'" in desc.lower())
        if (default_numeric or (metavar_num and not livre)) and not (
            "," in metavar or "..." in metavar
        ):
            kind = "number"
        elif re.match(r"^\d+(\.\.\.\d+)?$", metavar_inner):
            kind = "number"

    if key in ACTIONS or (not metavar and desc.lower().startswith(("print ", "show "))):
        kind = "action"

    # o default e a env ja vem dentro da descricao: tira de la (ficam nas badges)
    if mdef is not None:
        desc = re.sub(r"\(default:.*?\)", "", desc, flags=re.S)
    if env:
        desc = re.sub(r"\(env:\s*[A-Z0-9_]+\)", "", desc)
    desc = re.sub(r"[ \t]+\n", "\n", desc).strip()

    return {
        "key": key,
        "longs": longs,
        "shorts": shorts,
        "metavar": metavar,
        "desc": desc,
        "default": default,
        "env": env,
        "section": section,
        "tab": tab_for(section, key),
        "kind": kind,
        "choices": choices,
        "neg": neg,
        "order": order,
    }


def is_num(s):
    try:
        float(str(s).strip().strip('"').strip("'").rstrip(","))
        return True
    except Exception:  # noqa: BLE001
        return False


def tab_for(section, key):
    k = key
    if section == "sampling params":
        return "amostragem"
    if (
        k.startswith("spec-")
        or "draft" in k
        or k.startswith("ngram")
        or k.startswith("spec")
    ):
        return "spec"
    if k in SERVIDOR_KEYS:
        return "servidor"
    if any(k.startswith(p) or k == p for p in GPU_KEYS):
        return "gpu"
    if any(k.startswith(p) or k == p for p in CPU_KEYS) and not k.startswith("threads-http"):
        if k.startswith("threads") and k != "threads":
            pass
        return "cpu"
    if any(k.startswith(p) or k == p for p in CONTEXTO_KEYS):
        return "contexto"
    if any(k.startswith(p) or k == p for p in MODELO_KEYS):
        return "modelo"
    if any(k.startswith(p) or k == p for p in LOG_KEYS):
        return "log"
    return "geral"


# ---------------------------------------------------------------- config.ini

def parse_ini(text):
    """devolve (chaves_globais, secoes_nomeadas_brutas)"""
    glob = {}
    named = []
    cur = "*"
    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            if cur == "*":
                glob.setdefault("__raw__", []).append(line)
            else:
                named.append(line)
            continue
        m = re.match(r"^\[([^\]]*)\]$", s)
        if m:
            cur = m.group(1).strip() or "*"
            if cur == "*" or cur == "default":
                cur = "*"
                glob.setdefault("__raw__", []).append(line)
            else:
                named.append(line)
            continue
        mk = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$", s)
        if mk and cur == "*":
            glob[mk.group(1)] = mk.group(2).strip()
        else:
            named.append(line)
    glob.pop("__raw__", None)
    return glob, named


def validate_ini(text):
    erros = []
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        if re.match(r"^\[[^\]]*\]$", s):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*=\s*.*$", s):
            continue
        erros.append("linha %d invalida: %s" % (n, s[:70]))
    return erros


def validar_caminho(raw):
    """caminho absoluto (ou com ~) terminado em .ini -> Path normalizado.

    So valida onde o SITE vai ler/gravar; nao tem relacao com o caminho que o
    llama-server usa pro config dele.
    """
    s = str(raw or "").strip()
    if not s:
        raise ValueError("caminho vazio")
    if "\x00" in s:
        raise ValueError("caminho invalido (caractere nulo)")
    s = os.path.expanduser(s)
    if not os.path.isabs(s):
        raise ValueError("o caminho precisa ser absoluto (comecar com / ou ~)")
    p = Path(s)
    if ".." in p.parts:
        raise ValueError("o caminho nao pode conter ..")
    if p.suffix.lower() != ".ini":
        raise ValueError("o arquivo precisa terminar em .ini")
    if p.is_dir():
        raise ValueError("esse caminho e um diretorio, nao um arquivo")
    return p


def destino_do(raw):
    """parametro vazio/ausente -> caminho padrao"""
    if raw is None or str(raw).strip() == "":
        return CONFIG_PATH
    return validar_caminho(raw)


def write_config(text, known_keys, destino):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destino.exists():
        backup = destino.with_suffix(".ini.bak")
        shutil.copy2(destino, backup)
    tmp = destino.with_name("." + destino.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, destino)

    glob, _ = parse_ini(text)
    unknown = sorted(k for k in glob if k not in known_keys)
    return unknown, (str(backup) if backup else None)


# ---------------------------------------------------------------- http

_CACHE = {"flags": None, "binary": ""}


class Handler(BaseHTTPRequestHandler):
    server_version = "duway"

    # -- helpers ----------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    # -- rotas ------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, (ROOT / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if path == "/api/flags":
            return self.api_flags()
        if path == "/api/config":
            qs = parse_qs(urlparse(self.path).query)
            return self.api_config_get(qs.get("path", [""])[0])
        return self._json({"erro": "rota desconhecida"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/config":
            return self.api_config_post()
        return self._json({"erro": "rota desconhecida"}, 404)

    # -- apis -------------------------------------------------------------
    def _flags(self):
        binary = find_llama_bin()
        if not binary:
            raise RuntimeError(
                "llama-server nao encontrado (use LLAMA_BIN=/caminho/llama-server)"
            )
        if _CACHE["flags"] is None or _CACHE["binary"] != binary:
            flags, sections = parse_flags(binary)
            _CACHE["flags"] = (flags, sections)
            _CACHE["binary"] = binary
        return _CACHE["flags"]

    def api_flags(self):
        try:
            flags, sections = self._flags()
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": str(e)}, 500)
        known = set()
        for f in flags:
            known.add(f["key"])
            known.update(x.lstrip("-") for x in f["longs"])
            known.update(x.lstrip("-") for x in f["shorts"])
            if f["env"]:
                known.add(f["env"])
        self._json({
            "binary": _CACHE["binary"],
            "tabs": [{"id": i, "label": l} for i, l in TABS],
            "sections": sections,
            "flags": sorted(flags, key=lambda f: f["order"]),
            "known_keys": sorted(known),
        })

    def api_config_get(self, destino_raw=""):
        try:
            destino = destino_do(destino_raw)
        except ValueError as e:
            return self._json({"erro": str(e)}, 400)

        padrao = (destino == CONFIG_PATH)
        if not destino.exists():
            if not padrao:
                # caminho novo escolhido pelo site: nao crio nada por aqui,
                # o arquivo so nasce quando der APLICAR
                return self._json({
                    "path": str(destino),
                    "padrao": str(CONFIG_PATH),
                    "existe": False,
                    "criado": False,
                    "text": "",
                    "values": {},
                    "named": [],
                })
            try:
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(DEFAULT_CONFIG, encoding="utf-8")
                criado = True
            except Exception as e:  # noqa: BLE001
                return self._json({"erro": "nao criei o config: %s" % e}, 500)
        else:
            criado = False
        try:
            text = destino.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": "nao li o config: %s" % e}, 500)
        glob, named = parse_ini(text)
        self._json({
            "path": str(destino),
            "padrao": str(CONFIG_PATH),
            "existe": True,
            "criado": criado,
            "text": text,
            "values": glob,
            "named": named,
        })

    def api_config_post(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return self._json({"erro": "json invalido"}, 400)

        text = payload.get("text", "")
        if not isinstance(text, str):
            return self._json({"erro": "text ausente"}, 400)
        if not text.endswith("\n"):
            text += "\n"

        try:
            destino = destino_do(payload.get("path", ""))
        except ValueError as e:
            return self._json({"erro": str(e)}, 400)

        erros = validate_ini(text)
        if erros:
            return self._json({"erro": "config invalida", "detalhes": erros}, 400)

        try:
            flags, _ = self._flags()
            known = set()
            for f in flags:
                known.add(f["key"])
                known.update(x.lstrip("-") for x in f["longs"])
                known.update(x.lstrip("-") for x in f["shorts"])
                if f["env"]:
                    known.add(f["env"])
        except Exception:  # noqa: BLE001
            known = None

        try:
            unknown, backup = write_config(text, known or set(), destino)
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": "nao gravei: %s" % e}, 500)

        glob, _ = parse_ini(text)
        self._json({
            "ok": True,
            "path": str(destino),
            "padrao": str(CONFIG_PATH),
            "backup": backup,
            "count": len(glob),
            "unknown": unknown,
        })


def main():
    args = sys.argv[1:]
    global LISTEN_PORT
    if "--dump-flags" in args:
        binary = find_llama_bin()
        if not binary:
            print("llama-server nao encontrado", file=sys.stderr)
            return 1
        flags, sections = parse_flags(binary)
        print(json.dumps({"sections": sections, "flags": flags},
                         ensure_ascii=False, indent=2))
        return 0
    if "--porta" in args:
        LISTEN_PORT = int(args[args.index("--porta") + 1])
    if "--config" in args:  # soh cria/reescreve o config padrao
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")
        print("config padrao escrito em %s" % CONFIG_PATH)
        return 0

    binary = find_llama_bin()
    print("duway  ->  http://127.0.0.1:%d" % LISTEN_PORT)
    print("llama  ->  %s" % (binary or "NAO ENCONTRADO"))
    print("config ->  %s" % CONFIG_PATH)
    try:
        flags, _ = parse_flags(binary) if binary else ([], [])
        print("flags  ->  %d carregadas" % len(flags))
    except Exception as e:  # noqa: BLE001
        print("flags  ->  erro: %s" % e)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    except OSError as e:
        if e.errno in (98, 48, 10048):  # endereco em uso
            print(
                "a porta %d ja esta em uso.\n"
                "  -> tem outro duway rodando?  pkill -f server.py\n"
                "  -> ou rode noutro lugar:     python3 server.py --porta 8788"
                % LISTEN_PORT,
                file=sys.stderr,
            )
            return 1
        raise
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nate logo")
