"""Gerador: o agente que responde à pergunta usando só o contexto selecionado."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from agno.agent import Agent
from agno.run.base import RunStatus

from rag_knapsack.cache import CacheJson, chave
from rag_knapsack.dados import Chunk
from rag_knapsack.llm import ErroDeModelo, Gasto, modelo_openai
from rag_knapsack.metricas import RESPOSTA_DESCONHECIDA

# Prompts em inglês: o HotpotQA é em inglês e EM/F1 comparam com o gabarito palavra por palavra.
DESCRICAO = "You answer questions using only the Wikipedia passages provided by the user."
INSTRUCOES = [
    "Use only the passages; do not rely on prior knowledge.",
    "Reply with the shortest possible answer (an entity, date, number or short phrase), in English, "
    "with no explanation.",
    "For yes/no questions, reply exactly 'yes' or 'no'.",
    f"If the passages do not contain the answer, reply exactly '{RESPOSTA_DESCONHECIDA}'.",
]


@dataclass(frozen=True)
class RespostaGerada:
    texto: str
    tokens_entrada: int
    tokens_saida: int  # inclui os tokens de raciocínio, que a OpenAI cobra como saída
    tokens_raciocinio: int
    duracao_s: float | None


def montar_prompt(pergunta: str, chunks: Sequence[Chunk]) -> str:
    passagens = "\n\n".join(f"[{i}] {chunk.bloco}" for i, chunk in enumerate(chunks, start=1))
    return f"Passages:\n\n{passagens or '(none)'}\n\nQuestion: {pergunta}"


class Gerador:
    def __init__(self, modelo: str, esforco: str, cache: CacheJson | None = None) -> None:
        self.modelo = modelo
        self.esforco = esforco
        self.cache = cache
        self.gasto = Gasto()

    def responder(self, pergunta: str, chunks: Sequence[Chunk]) -> RespostaGerada:
        prompt = montar_prompt(pergunta, chunks)
        chave_cache = chave(self.modelo, self.esforco, DESCRICAO, INSTRUCOES, prompt)
        if self.cache is not None and chave_cache in self.cache:
            return RespostaGerada(**self.cache.get(chave_cache))
        # Um agente por chamada: o Agent guarda estado de sessão e as chamadas rodam em paralelo.
        agente = Agent(
            model=modelo_openai(self.modelo, self.esforco),
            description=DESCRICAO,
            instructions=INSTRUCOES,
            telemetry=False,
        )
        saida = agente.run(prompt)
        metricas = saida.metrics
        if metricas:  # tokens cobrados contam mesmo se a resposta vier inválida
            self.gasto.registrar(metricas.input_tokens, metricas.output_tokens)
        if saida.status != RunStatus.completed or not isinstance(saida.content, str) or not saida.content.strip():
            raise ErroDeModelo(f"{self.modelo} não respondeu: {saida.content!r}")
        resposta = RespostaGerada(
            texto=saida.content.strip(),
            tokens_entrada=metricas.input_tokens if metricas else 0,
            tokens_saida=metricas.output_tokens if metricas else 0,
            tokens_raciocinio=metricas.reasoning_tokens if metricas else 0,
            duracao_s=metricas.duration if metricas else None,
        )
        if self.cache is not None:
            self.cache.set(chave_cache, asdict(resposta))
        return resposta
