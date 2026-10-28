"""HotpotQA (dev, cenário *distractor*): download, leitura e amostragem.

No cenário distractor cada pergunta já vem com 10 parágrafos da Wikipedia: os 2 que
sustentam a resposta (*supporting facts*) e 8 distratores. Cada parágrafo vira um chunk
candidato. Dataset: Yang et al. (2018), licença CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rag_knapsack.config import Config

URL_LINHAS = "https://datasets-server.huggingface.co/rows"
PARAMETROS_DATASET = {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"}
LINHAS_POR_REQUISICAO = 100  # máximo aceito pela API
PAUSA_ENTRE_REQUISICOES_S = 1.0  # a API devolve 429 se as páginas forem pedidas em rajada


@dataclass(frozen=True)
class Chunk:
    id: str
    titulo: str
    texto: str
    de_suporte: bool = False
    tokens: int = 0  # peso w_i, preenchido pelo recuperador
    score: float = 0.0  # valor v_i, preenchido pelo recuperador

    @property
    def bloco(self) -> str:
        """Como o chunk aparece no prompt; é este texto que é embutido e contado em tokens."""
        return f"Title: {self.titulo}\n{self.texto}"


@dataclass(frozen=True)
class Exemplo:
    id: str
    pergunta: str
    resposta: str
    tipo: str  # bridge | comparison
    nivel: str  # easy | medium | hard
    chunks: tuple[Chunk, ...]

    @property
    def ids_suporte(self) -> frozenset[str]:
        return frozenset(c.id for c in self.chunks if c.de_suporte)

    def com_chunks(self, chunks: tuple[Chunk, ...]) -> Exemplo:
        return replace(self, chunks=chunks)


def exemplo_de_linha(linha: dict) -> Exemplo:
    """Converte uma linha no formato do Hugging Face em um `Exemplo`."""
    titulos_suporte = set(linha["supporting_facts"]["title"])
    chunks: list[Chunk] = []
    vistos: set[str] = set()
    for titulo, sentencas in zip(linha["context"]["title"], linha["context"]["sentences"], strict=True):
        id_chunk = titulo
        while id_chunk in vistos:
            id_chunk += "'"
        vistos.add(id_chunk)
        # As sentenças do HotpotQA já trazem o espaço inicial; juntá-las reconstrói o parágrafo.
        texto = "".join(sentencas).strip()
        chunks.append(Chunk(id=id_chunk, titulo=titulo, texto=texto, de_suporte=titulo in titulos_suporte))
    return Exemplo(
        id=linha["id"],
        pergunta=linha["question"],
        resposta=linha["answer"],
        tipo=linha["type"],
        nivel=linha["level"],
        chunks=tuple(chunks),
    )


def carregar_exemplos(caminho: Path, limite: int | None = None) -> list[Exemplo]:
    exemplos: list[Exemplo] = []
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            if linha.strip():
                exemplos.append(exemplo_de_linha(json.loads(linha)))
            if limite is not None and len(exemplos) >= limite:
                break
    return exemplos


def _baixar_json(url: str, tentativas: int = 6) -> dict:
    ultimo_erro: Exception | None = None
    for tentativa in range(tentativas):
        espera = 5 * 2**tentativa
        try:
            requisicao = Request(url, headers={"User-Agent": "rag-knapsack (pesquisa academica)"})
            with urlopen(requisicao, timeout=60) as resposta:
                return json.load(resposta)
        except HTTPError as erro:
            if erro.code != 429 and erro.code < 500:
                raise
            ultimo_erro = erro
            retry_after = erro.headers.get("Retry-After", "")
            if retry_after.isdigit():
                espera = int(retry_after)
        except URLError as erro:
            ultimo_erro = erro
        print(f"\n{ultimo_erro}; nova tentativa em {espera}s", file=sys.stderr)
        time.sleep(espera)
    raise RuntimeError(f"Falha ao baixar {url}: {ultimo_erro}")


def _linhas_completas(caminho: Path) -> int:
    """Conta (e preserva) as linhas íntegras de um download interrompido."""
    if not caminho.exists():
        return 0
    linhas = caminho.read_text(encoding="utf-8").splitlines(keepends=True)
    if linhas and not linhas[-1].endswith("\n"):
        linhas.pop()
        caminho.write_text("".join(linhas), encoding="utf-8")
    return len(linhas)


def baixar_hotpotqa(destino: Path, limite: int | None = None) -> int:
    """Baixa o dev distractor pela API de linhas do Hugging Face e grava em JSONL.

    Se o download for interrompido, rodar de novo continua de onde parou.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".parcial")
    gravadas, total = _linhas_completas(temporario), limite
    with temporario.open("a", encoding="utf-8") as arquivo:
        while total is None or gravadas < total:
            quantidade = LINHAS_POR_REQUISICAO if total is None else min(LINHAS_POR_REQUISICAO, total - gravadas)
            parametros = {**PARAMETROS_DATASET, "offset": gravadas, "length": quantidade}
            pagina = _baixar_json(f"{URL_LINHAS}?{urlencode(parametros)}")
            if total is None:
                total = pagina["num_rows_total"]
            for item in pagina["rows"]:
                arquivo.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
            arquivo.flush()
            gravadas += len(pagina["rows"])
            print(f"\r{gravadas}/{total} perguntas", end="", file=sys.stderr, flush=True)
            if not pagina["rows"]:
                break
            time.sleep(PAUSA_ENTRE_REQUISICOES_S)
    print(file=sys.stderr)
    temporario.replace(destino)
    return gravadas


def gerar_amostra(origem: Path, destino: Path, n: int, semente: int) -> None:
    linhas = [linha for linha in origem.read_text(encoding="utf-8").splitlines() if linha.strip()]
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(random.Random(semente).sample(linhas, n)) + "\n", encoding="utf-8")


def caminho_dev(cfg: Config) -> Path:
    return cfg.dir_dados / "hotpotqa" / "dev_distractor.jsonl"


CAMINHO_AMOSTRA = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "hotpotqa_dev_amostra.jsonl"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="comando", required=True)
    p_baixar = sub.add_parser("baixar", help="baixa o dev distractor (7.405 perguntas)")
    p_baixar.add_argument("--limite", type=int, default=None, help="baixa só as primeiras N perguntas")
    p_amostra = sub.add_parser("amostra", help="sorteia a amostra fixa usada nos testes")
    p_amostra.add_argument("--n", type=int, default=10)
    p_amostra.add_argument("--semente", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = Config.do_ambiente()
    if args.comando == "baixar":
        n = baixar_hotpotqa(caminho_dev(cfg), args.limite)
        print(f"{n} perguntas gravadas em {caminho_dev(cfg)}")
    else:
        if not caminho_dev(cfg).exists():
            sys.exit("Baixe o dataset primeiro: uv run task dados")
        gerar_amostra(caminho_dev(cfg), CAMINHO_AMOSTRA, args.n, args.semente)
        print(f"Amostra de {args.n} perguntas (semente {args.semente}) gravada em {CAMINHO_AMOSTRA}")


if __name__ == "__main__":
    main()
