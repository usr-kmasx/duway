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

import atexit
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import urllib.request
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

# ------------------------------------------------------------------- mcp
# servidores MCP = processos filhos do server.py, falando json-rpc por stdio.
# tudo que eles baixam fica DENTRO da pasta do projeto (nada no ~):
#   cache/uv  cache/pip  cache/pw  cache/tmp  cache/wdm  .venv

MCP_PATH = Path(
    os.environ.get("XDG_CONFIG_HOME") or (HOME / ".config")
) / "duway" / "mcp.json"
VENV_DIR = ROOT / ".venv"
CACHE_DIR = ROOT / "cache"

MCPS = {}          # nome -> registro {spec, proc, status, erro, tools, ...}
MCPS_LOCK = threading.Lock()


def _mcp_env(extra=None):
    env = dict(os.environ)
    for d in ("uv", "pip", "pw", "tmp", "wdm", "npm"):
        (CACHE_DIR / d).mkdir(parents=True, exist_ok=True)
    env["UV_CACHE_DIR"] = str(CACHE_DIR / "uv")
    env["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(CACHE_DIR / "pw")
    env["TMPDIR"] = str(CACHE_DIR / "tmp")
    env["WDM_CACHE"] = str(CACHE_DIR / "wdm")
    env["WDM_LOCAL"] = "1"
    # npm/npx: cache e resto contidos no projeto (nada em ~/.npm)
    env["npm_config_cache"] = str(CACHE_DIR / "npm")
    env["npm_config_update_notifier"] = "false"
    env["npm_config_audit"] = "false"
    env["npm_config_fund"] = "false"
    lb = str(HOME / ".local" / "bin")
    if lb not in env.get("PATH", "").split(":"):
        env["PATH"] = lb + ":" + env.get("PATH", "")
    rb = str(ROOT / "bin")   # shim google-chrome -> chromium (só nos processos MCP)
    if rb not in env.get("PATH", "").split(":"):
        env["PATH"] = rb + ":" + env.get("PATH", "")
    nb = str(ROOT / "node" / "bin")   # node portátil (npx) — baixado na 1a instalação npm
    if (ROOT / "node").exists() and nb not in env.get("PATH", "").split(":"):
        env["PATH"] = nb + ":" + env.get("PATH", "")
    for k, v in (extra or {}).items():
        env[str(k)] = str(v)
    return env


def _mcp_validar(text):
    """devolve (servers, erro) — valida a forma do mcp.json"""
    try:
        doc = json.loads(text or "{}")
    except Exception as e:  # noqa: BLE001
        return None, "json invalido: %s" % e
    if not isinstance(doc, dict) or not isinstance(doc.get("servers"), list):
        return None, 'esperado um objeto {"servers": [...]}'
    vistos = set()
    for i, s in enumerate(doc["servers"]):
        if not isinstance(s, dict):
            return None, "servers[%d] nao e objeto" % i
        nome, cmd = s.get("nome"), s.get("cmd")
        if not isinstance(nome, str) or not nome.strip():
            return None, "servers[%d]: nome vazio" % i
        if nome in vistos:
            return None, "nome repetido: %s" % nome
        vistos.add(nome)
        if not isinstance(cmd, str) or not cmd.strip():
            return None, "%s: cmd vazio" % nome
        if not isinstance(s.get("args", []), list) or any(
            not isinstance(a, str) for a in s.get("args", [])
        ):
            return None, "%s: args precisa ser lista de texto" % nome
        if not isinstance(s.get("env", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in s.get("env", {}).items()
        ):
            return None, "%s: env precisa ser texto=texto" % nome
        if not isinstance(s.get("ativo", True), bool):
            return None, "%s: ativo precisa ser true/false" % nome
        if not isinstance(s.get("instalar", []), list) or any(
            not isinstance(p, str) for p in s.get("instalar", [])
        ):
            return None, "%s: instalar precisa ser lista de texto" % nome
    return doc["servers"], None


def _mcp_gravar(text):
    MCP_PATH.parent.mkdir(parents=True, exist_ok=True)
    bak = Path(str(MCP_PATH) + ".bak")
    if MCP_PATH.exists():
        shutil.copy2(MCP_PATH, bak)
    tmp = Path(str(MCP_PATH) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, MCP_PATH)
    return bak


def _mcp_novo(spec):
    return {
        "spec": spec, "proc": None, "status": "parado", "erro": "",
        "tools": [], "seq": 0, "pend": {}, "stderr": [],
        "lock": threading.Lock(),
    }


def _mcp_nome_de(tok, runtime):
    base = tok
    if runtime == "uvx" and (base.startswith("git+") or "://" in base
                             or "/" in base):    # git -> ultimo pedaco da url
        base = base.rstrip("/").split("/")[-1]
        if base.endswith(".git"):
            base = base[:-4]
        base = base.split("#")[0].split("@")[0]
    elif base.startswith("@"):                   # npm: @escopo/pacote@1.0
        base = base[1:].split("@")[0].replace("/", "-")
    else:                                        # pypi: pacote==1.0
        base = base.split("@")[0].split("==")[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-").lower()
    return base or "mcp"


def _mcp_parse_entrada(txt):
    """'colo o link' -> (cmd, args, nome); prefixo npx:/uvx: forca o runtime"""
    t = (txt or "").strip()
    if not t:
        raise ValueError("cole um link ou o nome de um pacote")
    forcado = None
    if t.startswith("npx:") or t.startswith("uvx:"):
        forcado, _, t = t.partition(":")
        t = t.strip()
    elif re.match(r"^(npx|uvx)(\s|$)", t):       # linha colada inteira
        partes = t.split()
        forcado, t = partes[0], " ".join(partes[1:])
    if not t:
        raise ValueError("faltou o link/pacote depois de " + (forcado or ""))
    eh_git = ("://" in t or t.startswith(("git+", "github.com/", "gitlab.com/",
                                          "bitbucket.org/"))
              or bool(re.match(r"^[\w.-]+/[\w.-]+$", t)))
    runtime = forcado or ("npx" if t.startswith("@") and not eh_git else "uvx")
    if runtime == "npx":
        toks = [p for p in t.split() if p not in ("-y", "--yes")]
        if not toks:
            raise ValueError("faltou o nome do pacote npm")
        cmd, args = "npx", ["-y"] + toks
    else:
        partes = t.split()
        # linha de docs colada com flags: uvx --from X --with Y cmd
        tem_flag = any(p in ("--from", "--with")
                       or p.startswith(("--from=", "--with=")) for p in partes)
        pos, i = [], 0
        while i < len(partes):
            p = partes[i]
            if p in ("--from", "--with"):
                i += 2
                continue
            if p.startswith(("--from=", "--with=")) or p.startswith("-"):
                i += 1
                continue
            pos.append(p)
            i += 1
        if tem_flag and not pos:
            raise ValueError(
                "--with/--from veio sem o comando que deve rodar — cole só o "
                "link do repositório, ou a linha completa com o comando depois "
                "da flag")
        spec = t
        if re.match(r"^[A-Za-z][\w+.-]*://", spec) and not spec.startswith("git+"):
            spec = "git+" + spec
        elif spec.startswith(("github.com/", "gitlab.com/", "bitbucket.org/")):
            spec = "git+https://" + spec
        elif re.match(r"^[\w.-]+/[\w.-]+$", spec):   # usuario/repositorio
            spec = "git+https://github.com/" + spec
        toks = spec.split()
        if not toks:
            raise ValueError("faltou o link/pacote")
        cmd, args = "uvx", toks
        return cmd, args, _mcp_nome_de(pos[0] if pos else toks[0], runtime)
    return cmd, args, _mcp_nome_de(toks[0], runtime)


def _mcp_node():
    """node portatil DENTRO do projeto (baixado so na 1a instalacao npm)"""
    b = ROOT / "node" / "bin"
    if (b / "npx").exists():
        return b
    maq = platform.machine().lower()
    arq = {"x86_64": "x64", "amd64": "x64",
           "aarch64": "arm64", "arm64": "arm64"}.get(maq)
    if not arq:
        raise ValueError("arquitetura %s sem node portatil" % maq)
    with urllib.request.urlopen(
            "https://nodejs.org/dist/index.json", timeout=30) as r:
        idx = json.loads(r.read().decode("utf-8"))
    ver = next((v["version"] for v in idx if v.get("lts")), None)
    if not ver:
        raise ValueError("nao achei a versao LTS do node")
    url = ("https://nodejs.org/dist/%s/node-%s-linux-%s.tar.xz"
           % (ver, ver, arq))
    td = CACHE_DIR / "tmp"
    td.mkdir(parents=True, exist_ok=True)
    tar = td / ("node-%s.tar.xz" % ver)
    with urllib.request.urlopen(url, timeout=180) as r, open(tar, "wb") as f:
        while True:
            bloco = r.read(1 << 20)
            if not bloco:
                break
            f.write(bloco)
    extra = td / ("node-ex-%s" % ver)
    shutil.rmtree(extra, ignore_errors=True)
    extra.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        try:
            tf.extractall(extra, filter="data")   # python >=3.12
        except TypeError:                          # python antigo: sem filter
            tf.extractall(extra)
    orig = extra / ("node-%s-linux-%s" % (ver, arq))
    if not (orig / "bin" / "npx").exists():
        raise ValueError("tarball do node veio estranho (sem bin/npx)")
    shutil.rmtree(ROOT / "node", ignore_errors=True)
    os.rename(orig, ROOT / "node")
    tar.unlink(missing_ok=True)
    shutil.rmtree(extra, ignore_errors=True)
    return b


def _mcp_chamar(m, metodo, params, timeout=60):
    """json-rpc por stdio: escreve a requisicao, espera a resposta com o id"""
    with m["lock"]:
        if not m["proc"] or m["proc"].poll() is not None:
            raise RuntimeError("processo nao esta rodando")
        m["seq"] += 1
        rid = m["seq"]
        fila = queue.Queue()
        m["pend"][rid] = fila
        msg = json.dumps({"jsonrpc": "2.0", "id": rid,
                          "method": metodo, "params": params}) + "\n"
        try:
            m["proc"].stdin.write(msg)
            m["proc"].stdin.flush()
        except Exception as e:  # noqa: BLE001
            m["pend"].pop(rid, None)
            raise RuntimeError("sem stdin: %s" % e)
        try:
            resp = fila.get(timeout=timeout)
        except queue.Empty:
            m["pend"].pop(rid, None)
            raise TimeoutError("%s nao respondeu em %ds" % (metodo, timeout))
        if resp is None:
            raise RuntimeError("processo morreu durante %s%s" % (
                metodo,
                (" — " + " | ".join(m["stderr"][-3:])) if m["stderr"] else "",
            ))
        if "error" in resp:
            raise RuntimeError(str(resp["error"]))
        return resp.get("result")


def _mcp_stdout(m):
    for line in m["proc"].stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(msg, dict):
            continue
        rid = msg.get("id")
        fila = m["pend"].pop(rid, None) if rid is not None else None
        if fila is not None:
            fila.put(msg)
    # EOF: processo acabou — acorda quem estava esperando
    for fila in m["pend"].values():
        fila.put(None)
    m["pend"].clear()
    if m["proc"] is not None:  # morreu sozinho (nao fui eu que matei)
        m["status"] = "erro"
        m["erro"] = m["erro"] or "processo morreu"


def _mcp_stderr(m):
    for line in m["proc"].stderr:
        line = line.rstrip()
        if line:
            m["stderr"].append(line)
            del m["stderr"][:-100]   # guarda mais linhas: crash do chrome cabe


def _mcp_matar(m):
    p = m.get("proc")
    m["proc"] = None
    for fila in m["pend"].values():
        fila.put(None)
    m["pend"].clear()
    if p:
        # grupo de processos (uvx -> python do mcp): mata tudo, nao só o pai
        try:
            pgid = os.getpgid(p.pid)
        except Exception:  # noqa: BLE001
            pgid = None
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
        if pgid:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        try:
            p.wait(timeout=3)
        except Exception:  # noqa: BLE001
            if pgid:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
    if not m["erro"]:
        m["status"] = "parado"


def _mcp_subir(m):
    """sobe o processo + handshake (initialize/tools/list) — bloqueante,
    chamado FORA do MCPS_LOCK pra nao travar o GET /api/mcp"""
    spec = m["spec"]
    if m["proc"]:
        _mcp_matar(m)
    m["status"] = "subindo"
    m["erro"] = ""
    try:
        p = subprocess.Popen(
            [spec["cmd"]] + list(spec.get("args", [])),
            cwd=str(ROOT),
            env=_mcp_env(spec.get("env")),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True,   # grupo proprio: o killpg mata o uvx + filho
        )
    except Exception as e:  # noqa: BLE001
        m["status"] = "erro"
        m["erro"] = "nao subiu: %s" % e
        return
    m["proc"] = p
    threading.Thread(target=_mcp_stdout, args=(m,), daemon=True).start()
    threading.Thread(target=_mcp_stderr, args=(m,), daemon=True).start()
    try:
        _mcp_chamar(m, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "duway", "version": "1"},
        }, timeout=120)   # 1a vez: o uvx baixa o pacote nesse meio-tempo
        p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        p.stdin.flush()
        res = _mcp_chamar(m, "tools/list", {}, timeout=30)
        m["tools"] = (res or {}).get("tools", []) if isinstance(res, dict) else []
        m["status"] = "rodando"
    except Exception as e:  # noqa: BLE001
        m["erro"] = str(e)
        _mcp_matar(m)
        m["status"] = "erro"


def _mcp_aplicar(servers):
    """sincroniza MCPS com a lista nova do arquivo; devolve status por nome"""
    subir = []
    with MCPS_LOCK:
        alvo = {s["nome"]: s for s in servers}
        for nome in list(MCPS.keys()):     # saiu / desligou / mudou
            m = MCPS[nome]
            s = alvo.get(nome)
            mudou = (
                s is None or not s.get("ativo", True)
                or s.get("cmd") != m["spec"].get("cmd")
                or list(s.get("args", [])) != list(m["spec"].get("args", []))
                or (s.get("env") or {}) != (m["spec"].get("env") or {})
            )
            if mudou:
                _mcp_matar(m)
                if s is None:
                    MCPS.pop(nome, None)
                else:
                    m["spec"] = s
                    m["tools"] = []
                    m["erro"] = ""
        for nome, s in alvo.items():       # entrou / atualizou spec
            if nome not in MCPS:
                MCPS[nome] = _mcp_novo(s)
            else:
                MCPS[nome]["spec"] = s
        for nome, s in alvo.items():       # sobe os ativos
            if not s.get("ativo", True):
                continue
            m = MCPS[nome]
            if m["proc"] is None or m["status"] == "erro":
                subir.append(m)
    for m in subir:
        _mcp_subir(m)
    return _mcp_status(servers)


def _mcp_status(servers=None):
    out = []
    with MCPS_LOCK:
        if servers is None:
            servers = [m["spec"] for m in MCPS.values()]
        for s in servers:
            m = MCPS.get(s.get("nome", ""), {})
            out.append({
                "nome": s.get("nome", ""),
                "cmd": s.get("cmd", ""),
                "args": s.get("args", []),
                "env": s.get("env", {}),
                "instalar": s.get("instalar", []),
                "ativo": bool(s.get("ativo", True)),
                "status": m.get("status", "parado"),
                "erro": m.get("erro", ""),
                "log": m.get("stderr", [])[-100:],  # últimas linhas do stderr
                "tools": [
                    {"name": t.get("name", ""),
                     "description": t.get("description", "")}
                    for t in m.get("tools", [])
                ],
            })
    return out


def _mcp_boot():
    try:
        if not MCP_PATH.exists():
            return
        servers, erro = _mcp_validar(MCP_PATH.read_text(encoding="utf-8"))
        if erro:
            print("mcp   ->  %s" % erro)
            return
        if not servers:
            return
        if not VENV_DIR.exists():
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)],
                           check=True, timeout=120, env=_mcp_env())
        print("mcp   ->  %d servidor(es), %d ativo(s)" % (
            len(servers), len([s for s in servers if s.get("ativo", True)])))
        _mcp_aplicar(servers)
    except Exception as e:  # noqa: BLE001
        print("mcp   ->  erro: %s" % e)


def _mcp_desligar_tudo():
    with MCPS_LOCK:
        for m in MCPS.values():
            _mcp_matar(m)


atexit.register(_mcp_desligar_tudo)

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
        if path == "/api/mcp":
            return self.api_mcp_get()
        if path == "/api/tools":
            return self.api_tools_get()
        return self._json({"erro": "rota desconhecida"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/config":
            return self.api_config_post()
        if path == "/api/mcp":
            return self.api_mcp_post()
        if path == "/api/mcp/instalar":
            return self.api_mcp_instalar_post()
        if path == "/api/tool":
            return self.api_tool_post()
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


    # -- mcp ---------------------------------------------------------------
    def api_mcp_get(self):
        if not MCP_PATH.exists():
            return self._json({"path": str(MCP_PATH), "existe": False,
                               "servers": []})
        try:
            text = MCP_PATH.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": "nao li o mcp.json: %s" % e}, 500)
        servers, erro = _mcp_validar(text)
        if erro:
            return self._json({"erro": erro, "path": str(MCP_PATH)}, 400)
        self._json({"path": str(MCP_PATH), "existe": True,
                    "servers": _mcp_status(servers)})

    def api_mcp_post(self):
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

        servers, erro = _mcp_validar(text)
        if erro:
            return self._json({"erro": "mcp.json invalido",
                               "detalhes": [erro]}, 400)

        # .venv do projeto: nasce so na 1a gravacao (nunca toca o sistema)
        if not VENV_DIR.exists():
            try:
                VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [sys.executable, "-m", "venv", str(VENV_DIR)],
                    check=True, timeout=180, env=_mcp_env(),
                )
            except Exception as e:  # noqa: BLE001
                return self._json({"erro": "nao criei o .venv: %s" % e}, 500)

        # pacotes que o mcp pediu (campo instalar) — so pra dentro do .venv
        for s in servers:
            pkgs = s.get("instalar") or []
            if not pkgs:
                continue
            uv = shutil.which("uv")
            cmd = (([uv, "pip", "install", "-p", str(VENV_DIR)] if uv else
                    [str(VENV_DIR / "bin" / "python"), "-m", "pip", "install"])
                   + list(pkgs))
            try:
                subprocess.run(cmd, check=True, timeout=600, env=_mcp_env())
            except Exception as e:  # noqa: BLE001
                return self._json({
                    "erro": "%s: falhou ao instalar %s (%s)"
                            % (s["nome"], " ".join(pkgs), e)}, 500)

        try:
            bak = _mcp_gravar(text)
            status = _mcp_aplicar(servers)
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": "nao gravei: %s" % e}, 500)

        self._json({"ok": True, "path": str(MCP_PATH), "backup": str(bak),
                    "servers": status})

    def api_mcp_instalar_post(self):
        """'colo o link' -> baixa, sobe, confere (tools/list) e SÓ grava se OK"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return self._json({"erro": "json invalido"}, 400)
        try:
            cmd, args, nome = _mcp_parse_entrada(payload.get("entrada", ""))
        except ValueError as e:
            return self._json({"erro": str(e)}, 400)

        texto = '{"servers": []}\n'
        if MCP_PATH.exists():
            try:
                texto = MCP_PATH.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                return self._json({"erro": "nao li o mcp.json: %s" % e}, 500)
        try:
            doc = json.loads(texto or "{}")
            servers = doc.get("servers") or []
            if not isinstance(servers, list):
                raise ValueError("servers nao e lista")
        except Exception as e:  # noqa: BLE001
            return self._json({"erro": "mcp.json quebrado (%s) — conserte "
                                       "na tela antes de instalar" % e}, 400)

        # mesmo cmd+args ja ta na lista? entao so devolve o status dele
        for s in servers:
            if s.get("cmd") == cmd and list(s.get("args") or []) == args:
                st = _mcp_status([s])[0]
                return self._json({"ok": True, "ja_instalado": True,
                                   "nome": s.get("nome"),
                                   "n_tools": len(st.get("tools") or []),
                                   "tools": st.get("tools") or [],
                                   "servers": _mcp_status()})

        base, n2 = nome, 1                       # nome repetido -> sufixa
        while any(s.get("nome") == nome for s in servers):
            n2 += 1
            nome = "%s-%d" % (base, n2)
        spec = {"nome": nome, "cmd": cmd, "args": args, "ativo": True}

        if cmd == "npx":
            try:
                _mcp_node()                      # node portatil (so na 1a vez)
            except Exception as e:  # noqa: BLE001
                return self._json({"erro": "nao baixei o node: %s" % e}, 500)

        # sonda SEM gravar: sobe o processo, initialize + tools/list
        def sonda(cmd_s, args_s):
            ms = _mcp_novo({"nome": nome, "cmd": cmd_s, "args": args_s,
                            "ativo": True})
            with MCPS_LOCK:
                MCPS[nome] = ms
            _mcp_subir(ms)
            return ms

        def fecha(ms):
            _mcp_matar(ms)
            with MCPS_LOCK:
                MCPS.pop(nome, None)

        m = sonda(cmd, args)
        aviso = ""

        def volta_pra_tras(motivo):
            log = list(m.get("stderr", []))[-30:]
            fecha(m)
            return self._json({"erro": motivo,
                               "detalhes": log or [m.get("erro", "") or "?"]},
                              422)

        if m["status"] != "rodando":
            texto = "\n".join(list(m.get("stderr", [])))
            # quando o repo nao e python o uv cita o dir do checkout no erro
            ck = re.search(r"(/[^\s]*?checkouts/\S+)", texto)
            dir_ck = ck.group(1).rstrip("\"',)") if ck else ""
            if cmd == "uvx" and "does not appear to be a Python project" in texto:
                url = next((a for a in args if "://" in a or a.startswith("git+")),
                           "")
                eh_js = bool(url) and dir_ck and os.path.isfile(
                    os.path.join(dir_ck, "package.json"))
                eh_go = bool(dir_ck) and os.path.isfile(
                    os.path.join(dir_ck, "go.mod"))
                if eh_js:
                    # escolha do usuario: repo JS -> tenta npx e avisa
                    try:
                        _mcp_node()
                    except Exception as e2:  # noqa: BLE001
                        fecha(m)
                        return self._json(
                            {"erro": "não baixei o node: %s" % e2}, 500)
                    fecha(m)
                    i = args.index(url)
                    extras = [a for a in args[i + 1:] if not a.startswith("-")]
                    cmd, args = "npx", ["-y", url] + extras
                    m = sonda(cmd, args)
                    if m["status"] == "rodando":
                        aviso = ("o repo não é Python (é Node) — instalei como "
                                 "npm/npx")
                    else:
                        return volta_pra_tras(
                            "%s não subiu nem como npm/npx" % nome)
                elif eh_go:
                    return volta_pra_tras(
                        "%s é escrito em Go (go.mod): aqui só rodam mcps "
                        "Python (uvx) e Node (npx), e ele não publica binário "
                        "— por link não dá pra instalar" % nome)
                else:
                    return volta_pra_tras(
                        "o repo de %s não é projeto Python nem Node (sem "
                        "pyproject/setup.py/package.json) — não dá pra rodar "
                        "por uvx/npx" % nome)
            elif any(k in texto for k in ("Resolved ", "Downloaded ",
                                          "Prepared ")):
                return volta_pra_tras(
                    "%s baixou os pacotes mas o processo não falou MCP "
                    "(handshake não completou) — o repo pode ser um cliente/"
                    "CLI em vez de um servidor; veja o log" % nome)
            else:
                return volta_pra_tras("%s não subiu" % nome)
        tools = list(m.get("tools", []))

        # dedupe de novo com o cmd final (uvx pode ter virado npx ali em cima)
        for s in servers:
            if s.get("cmd") == cmd and list(s.get("args") or []) == args:
                fecha(m)
                st = _mcp_status([s])[0]
                return self._json({"ok": True, "ja_instalado": True,
                                   "nome": s.get("nome"),
                                   "n_tools": len(st.get("tools") or []),
                                   "tools": st.get("tools") or [],
                                   "servers": _mcp_status()})

        servers.append(m["spec"])                # so aqui o arquivo e tocado
        novo, err = _mcp_validar(json.dumps({"servers": servers}))
        if err:
            return volta_pra_tras("config ficaria invalida: %s" % err)
        try:
            bak = _mcp_gravar(
                json.dumps({"servers": novo}, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            return volta_pra_tras("nao gravei: %s" % e)
        self._json({"ok": True, "path": str(MCP_PATH), "backup": str(bak),
                    "nome": nome, "n_tools": len(tools), "tools": tools,
                    "aviso": aviso, "servers": _mcp_status()})

    def api_tools_get(self):
        """tools de todos os MCP ativos e rodando (pro chat usar)"""
        out = []
        with MCPS_LOCK:
            for nome, m in MCPS.items():
                if m["proc"] is not None and m["status"] == "rodando":
                    out.append({"servidor": nome, "ativo": True,
                                "tools": m["tools"]})
        self._json({"servidores": out})

    def api_tool_post(self):
        """executa uma tool: {servidor, name, arguments} -> content do MCP"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            return self._json({"erro": "json invalido"}, 400)

        nome = payload.get("servidor", "")
        name = payload.get("name", "")
        if not isinstance(nome, str) or not isinstance(name, str) or not name:
            return self._json({"erro": "servidor/name ausentes"}, 400)

        with MCPS_LOCK:
            m = MCPS.get(nome)
            if m is None:
                return self._json({"erro": "servidor desconhecido: %s" % nome},
                                  404)
            ref = m
        if ref["proc"] is None or ref["status"] != "rodando":
            return self._json({"erro": "servidor nao esta rodando: %s" % nome},
                              409)

        def chamar():
            return _mcp_chamar(ref, "tools/call", {
                "name": name,
                "arguments": payload.get("arguments") or {},
            }, timeout=180)

        # o web-search-mcp NUNCA recria o driver morto (bug dele: fica com a
        # referencia antiga; o search engole a excecao e devolve [] , entao o
        # "invalid session id" aparece so no stderr). Reinicia o processo mcp
        # (filho NOSSO, nunca o llama) e repete UMA vez.
        def morta(res=None, inicio=0):
            if res is not None and "invalid session id" in json.dumps(
                    res, ensure_ascii=False):
                return True
            log = ref.get("stderr", [])
            cauda = log[inicio:] if len(log) >= inicio else log[-30:]
            return any("invalid session" in l for l in cauda)

        inicio = len(ref.get("stderr", []))
        try:
            res = chamar()
            reniciar = morta(res, inicio)
        except Exception:  # noqa: BLE001
            reniciar = True
        if reniciar:
            _mcp_matar(ref)
            _mcp_subir(ref)
            try:
                res = chamar()
            except Exception as e:  # noqa: BLE001
                return self._json(
                    {"erro": "falhou ao chamar %s: %s" % (name, e)}, 500)
            self._json({"ok": True, "resultado": res, "reiniciado": True})
            return
        self._json({"ok": True, "resultado": res, "reiniciado": False})


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

    threading.Thread(target=_mcp_boot, daemon=True).start()  # sobe os ativos

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
