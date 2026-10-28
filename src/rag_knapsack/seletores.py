"""Módulo de seleção: escolhe quais chunks candidatos entram no contexto do LLM.

Todos os seletores implementam a mesma interface (`selecionar(chunks, orcamento)`), o que
permite comparar os algoritmos com os mesmos candidatos, scores e orçamento T.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from rag_knapsack.dados import Chunk


@dataclass(frozen=True)
class Selecao:
    algoritmo: str
    chunks: tuple[Chunk, ...]
    orcamento: int

    def __post_init__(self) -> None:
        if self.orcamento <= 0:
            raise ValueError(f"O orçamento T precisa ser positivo, não {self.orcamento}")

    @property
    def tokens_usados(self) -> int:
        return sum(chunk.tokens for chunk in self.chunks)

    @property
    def valor_total(self) -> float:
        """Objetivo da mochila: soma dos scores dos chunks escolhidos."""
        return sum(chunk.score for chunk in self.chunks)

    @property
    def ocupacao(self) -> float:
        return self.tokens_usados / self.orcamento

    @property
    def estourou_orcamento(self) -> bool:
        return self.tokens_usados > self.orcamento


class Seletor(Protocol):
    @property
    def nome(self) -> str: ...

    def selecionar(self, chunks: Sequence[Chunk], orcamento: int) -> Selecao: ...


def _na_ordem_do_prompt(escolhidos: Sequence[Chunk], candidatos: Sequence[Chunk]) -> tuple[Chunk, ...]:
    """Score decrescente; em empate, a ordem original dos candidatos.

    A posição no contexto afeta o LLM (Lost in the Middle), então todos os seletores
    entregam os chunks escolhidos na mesma ordem: só o conjunto muda entre algoritmos.
    """
    posicao = {id(chunk): i for i, chunk in enumerate(candidatos)}
    return tuple(sorted(escolhidos, key=lambda chunk: (-chunk.score, posicao[id(chunk)])))


@dataclass(frozen=True)
class TopK:
    """Top-K como normalmente usado em RAG: os k chunks de maior score.

    Não olha o tamanho dos chunks, então pode estourar o orçamento T; o estouro fica
    registrado na `Selecao`. Com `respeitar_orcamento=True` percorre os chunks por score e
    só inclui o que ainda cabe. A variante "topk_corte" do experimento usa `k=None` (sem
    limite de quantidade): enche T por ordem de score, como Greedy e Knapsack, que também
    não têm limite de k. É a base de comparação justa para eles.
    """

    k: int | None = 4
    respeitar_orcamento: bool = False

    @property
    def nome(self) -> str:
        return "topk_corte" if self.respeitar_orcamento else "topk"

    def selecionar(self, chunks: Sequence[Chunk], orcamento: int) -> Selecao:
        ordenados = _na_ordem_do_prompt(chunks, chunks)
        limite = len(ordenados) if self.k is None else self.k
        if not self.respeitar_orcamento:
            return Selecao(self.nome, ordenados[:limite], orcamento)
        escolhidos: list[Chunk] = []
        usados = 0
        for chunk in ordenados:
            if len(escolhidos) == limite:
                break
            if usados + chunk.tokens <= orcamento:
                escolhidos.append(chunk)
                usados += chunk.tokens
        return Selecao(self.nome, tuple(escolhidos), orcamento)


def _densidade(chunk: Chunk) -> float:
    return chunk.score / chunk.tokens if chunk.tokens else float("inf")


@dataclass(frozen=True)
class Greedy:
    """Guloso por densidade: maior score por token primeiro, pulando o que não cabe.

    O(n log n). Não garante o ótimo: pode perder até o valor de um chunk em relação ao
    Knapsack. (A garantia de metade do ótimo só vale para o guloso *modificado*, que
    compara a solução gulosa com o melhor chunk isolado.)
    """

    nome: ClassVar[str] = "greedy"

    def selecionar(self, chunks: Sequence[Chunk], orcamento: int) -> Selecao:
        escolhidos: list[Chunk] = []
        usados = 0
        for chunk in sorted(chunks, key=_densidade, reverse=True):
            # Score <= 0 só pioraria o objetivo; a DP também nunca o escolhe.
            if chunk.score > 0 and usados + chunk.tokens <= orcamento:
                escolhidos.append(chunk)
                usados += chunk.tokens
        return Selecao(self.nome, _na_ordem_do_prompt(escolhidos, chunks), orcamento)


@dataclass(frozen=True)
class Knapsack:
    """Mochila 0/1 exata por programação dinâmica, em O(n·T) de tempo e memória.

    melhor[t] guarda o maior valor que cabe em t tokens usando os chunks vistos até agora:
        melhor[t] = max(melhor[t], melhor[t - w_i] + v_i)
    O vetor é percorrido de T para baixo para que cada chunk entre no máximo uma vez, e
    `pegou[i][t]` registra a decisão para reconstruir quais chunks formam o ótimo.
    """

    nome: ClassVar[str] = "knapsack"

    def selecionar(self, chunks: Sequence[Chunk], orcamento: int) -> Selecao:
        itens = [chunk for chunk in chunks if chunk.score > 0 and chunk.tokens <= orcamento]
        melhor = [0.0] * (orcamento + 1)
        pegou = [bytearray(orcamento + 1) for _ in itens]
        for i, chunk in enumerate(itens):
            peso, valor = chunk.tokens, chunk.score
            for t in range(orcamento, peso - 1, -1):
                if melhor[t - peso] + valor > melhor[t]:
                    melhor[t] = melhor[t - peso] + valor
                    pegou[i][t] = 1
        escolhidos: list[Chunk] = []
        t = orcamento
        for i in range(len(itens) - 1, -1, -1):
            if pegou[i][t]:
                escolhidos.append(itens[i])
                t -= itens[i].tokens
        return Selecao(self.nome, _na_ordem_do_prompt(escolhidos, chunks), orcamento)


# Os três comparados na proposta e, por último, a base justa: Top-K que enche T sem limite de k.
ALGORITMOS = ("topk", "greedy", "knapsack", "topk_corte")


def criar_seletor(algoritmo: str, k: int = 4) -> Seletor:
    if algoritmo == "topk":
        return TopK(k=k)
    if algoritmo == "topk_corte":
        return TopK(k=None, respeitar_orcamento=True)
    if algoritmo == "greedy":
        return Greedy()
    if algoritmo == "knapsack":
        return Knapsack()
    raise ValueError(f"Algoritmo desconhecido: {algoritmo!r}. Opções: {', '.join(ALGORITMOS)}")
