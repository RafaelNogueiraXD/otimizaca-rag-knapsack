from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from rag_knapsack.cache import CacheJson
from rag_knapsack.config import Config, tem_chave_openai
from rag_knapsack.dados import CAMINHO_AMOSTRA, Chunk, Exemplo, carregar_exemplos
from rag_knapsack.evals import eval_desempenho
from rag_knapsack.experimento import executar, gravar_experimento
from rag_knapsack.gerador import Gerador, RespostaGerada
from rag_knapsack.juiz import Juiz, Veredito
from rag_knapsack.llm import Gasto
from rag_knapsack.metricas import RESPOSTA_DESCONHECIDA, normalizar_resposta
from rag_knapsack.pipeline import Resultado, gerar_respostas, julgar, selecionar
from rag_knapsack.recuperador import criar_recuperador
from rag_knapsack.seletores import ALGORITMOS, criar_seletor

ORCAMENTOS_SINTETICOS = (256, 512)


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.do_ambiente()


# --- Dados reais (embeddings de verdade) ------------------------------------------------


@pytest.fixture(scope="session")
def exemplos(cfg: Config) -> list[Exemplo]:
    """As 10 perguntas da amostra fixa, com score e tokens de cada chunk."""
    if cfg.precisa_chave_openai and not tem_chave_openai():
        pytest.skip("os embeddings da OpenAI exigem OPENAI_API_KEY (ou rode com EMBEDDER=fastembed)")
    recuperador = criar_recuperador(cfg)
    preparados = [recuperador.preparar(exemplo) for exemplo in carregar_exemplos(CAMINHO_AMOSTRA)]
    recuperador.salvar_cache()
    return preparados


@pytest.fixture(scope="session")
def selecoes(cfg: Config, exemplos: list[Exemplo]) -> dict[str, list[Resultado]]:
    return {
        algoritmo: selecionar(exemplos, criar_seletor(algoritmo, k=cfg.top_k), cfg.orcamento_tokens)
        for algoritmo in ALGORITMOS
    }


@pytest.fixture(scope="session")
def respostas(cfg: Config, selecoes: dict[str, list[Resultado]]) -> dict[str, list[Resultado]]:
    """Gera e julga uma vez só para todos os algoritmos (contextos iguais viram uma chamada)."""
    if not tem_chave_openai():
        pytest.skip("os testes com LLM exigem OPENAI_API_KEY")
    todos = [resultado for resultados in selecoes.values() for resultado in resultados]
    gerar_respostas(todos, Gerador(cfg.modelo_gerador, cfg.esforco_gerador, CacheJson()), workers=8)
    julgar(todos, Juiz(cfg.modelo_juiz, cfg.esforco_juiz, CacheJson()), workers=8)
    return selecoes


# --- Dublês sem rede, para testar evals e painel ----------------------------------------


class GeradorFalso:
    """Responde o gabarito quando todos os parágrafos de suporte estão no contexto."""

    def __init__(self, exemplos: list[Exemplo]) -> None:
        self.por_pergunta = {exemplo.pergunta: exemplo for exemplo in exemplos}
        self.chamadas = 0
        self.cache = CacheJson()
        self.gasto = Gasto()
        self._trava = threading.Lock()

    def responder(self, pergunta: str, chunks: list[Chunk]) -> RespostaGerada:
        with self._trava:
            self.chamadas += 1
        exemplo = self.por_pergunta[pergunta]
        completo = exemplo.ids_suporte <= {chunk.id for chunk in chunks}
        texto = exemplo.resposta if completo else RESPOSTA_DESCONHECIDA
        return RespostaGerada(texto, sum(c.tokens for c in chunks) + 60, 12, 8, 0.9)


class JuizFalso:
    def __init__(self) -> None:
        self.chamadas = 0
        self.cache = CacheJson()
        self.gasto = Gasto()
        self._trava = threading.Lock()

    def avaliar(self, pergunta: str, referencia: str, resposta: str) -> Veredito:
        with self._trava:
            self.chamadas += 1
        correto = normalizar_resposta(resposta) == normalizar_resposta(referencia)
        return Veredito(correto, "igual ao gabarito" if correto else "diferente do gabarito")


def _score_sintetico(exemplo: Exemplo, chunk: Chunk) -> float:
    ruido = int(hashlib.sha256(f"{exemplo.id}{chunk.id}".encode()).hexdigest(), 16) % 1000 / 1000
    return round(0.15 + 0.35 * ruido + (0.3 if chunk.de_suporte else 0.0), 4)


@pytest.fixture(scope="session")
def exemplos_sinteticos() -> list[Exemplo]:
    """A amostra com scores e pesos determinísticos, sem embeddings nem tokenizador."""
    return [
        exemplo.com_chunks(
            tuple(
                replace(chunk, tokens=max(1, len(chunk.bloco) // 4), score=_score_sintetico(exemplo, chunk))
                for chunk in exemplo.chunks
            )
        )
        for exemplo in carregar_exemplos(CAMINHO_AMOSTRA)
    ]


@pytest.fixture(scope="session")
def pasta_experimento(tmp_path_factory: pytest.TempPathFactory, cfg: Config, exemplos_sinteticos) -> Path:
    """Uma pasta no formato de `uv run task avaliar`, gerada com os dublês."""
    seletores = [criar_seletor(nome, k=4) for nome in ALGORITMOS]
    resultados = executar(
        exemplos_sinteticos, seletores, ORCAMENTOS_SINTETICOS, GeradorFalso(exemplos_sinteticos), JuizFalso(), 4
    )
    pasta = tmp_path_factory.mktemp("resultados") / "avaliacao-20260101-000000"
    gravar_experimento(
        pasta,
        [resultado.como_dict(cfg) for resultado in resultados],
        exemplos_sinteticos,
        eval_desempenho(seletores, exemplos_sinteticos, ORCAMENTOS_SINTETICOS, iteracoes=2),
        {
            "data": "2026-01-01T00:00:00",
            "fonte": CAMINHO_AMOSTRA.name,
            "perguntas": len(exemplos_sinteticos),
            "algoritmos": list(ALGORITMOS),
            "orcamentos": list(ORCAMENTOS_SINTETICOS),
            "k": 4,
            "modelo_gerador": "falso",
            "esforco_gerador": "-",
            "modelo_juiz": "falso",
            "esforco_juiz": "-",
            "embedder": "sintetico",
            "tokenizador": "len/4",
            "precos_usd_por_mtok": {"entrada": 0.1, "saida": 0.5},
            "gasto_api": {},
        },
    )
    return pasta
