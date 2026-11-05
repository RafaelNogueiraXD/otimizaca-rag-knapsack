"""Análise assintótica dos seletores: tempo, memória, operações e qualidade em função de n e T.

Exemplos:
    uv run task assintotica                         # 200 casos, n de 5 a 800, T de 128 a 8192
    uv run task assintotica --casos 30 --rapido     # grade menor, para conferir rápido

Cada caso parte de uma pergunta do HotpotQA. A instância de tamanho n tem os parágrafos da
própria pergunta e, se n passar deles, parágrafos reais de outras perguntas (com seus pesos e
scores reais), como se o recuperador tivesse devolvido mais candidatos. As instâncias de um
caso são aninhadas: a de n = 50 contém a de n = 20, então cada curva mostra o mesmo caso
crescendo.

Para cada (caso, algoritmo, n, T) são medidos:
    tempo        tempo de relógio de uma seleção (mediana de repetições, perf_counter_ns,
                 coletor de lixo desligado, processo fixo num núcleo)
    operações    passos executados por uma réplica instrumentada de cada algoritmo, que roda
                 os mesmos laços e ordenações do código em seletores.py e é conferida, em toda
                 medição, para escolher exatamente os mesmos chunks
    memória      pico de memória alocada (tracemalloc), em alguns casos por configuração
    qualidade    valor da seleção ÷ valor ótimo (Knapsack DP) no mesmo T

Varreduras:
    principal    os 4 seletores em toda a grade n × T
    forca_bruta  força bruta (2^n subconjuntos) e DP, com n pequeno
    cenarios     melhor, típico e pior caso de cada algoritmo (T fixo)
    escala       pesos e T multiplicados por s: mesma solução, custo da DP ∝ s (pseudo-polinomial)

A saída vai para resultados/assintotica-<data>/ e é lida por `uv run task painel-assintotico`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

from rag_knapsack.config import Config, inteiro_positivo, tem_chave_openai
from rag_knapsack.dados import CAMINHO_AMOSTRA, Chunk, Exemplo, caminho_dev, carregar_exemplos
from rag_knapsack.pipeline import gravar_jsonl
from rag_knapsack.recuperador import criar_recuperador
from rag_knapsack.seletores import ALGORITMOS, Knapsack, Selecao, Seletor, criar_seletor

REGIME_N, REGIME_T = 100, 2048  # entradas grandes, onde os expoentes são ajustados por padrão
N_MINIMO_FORCA_BRUTA = 12  # abaixo disso, os termos aditivos ainda pesam no ajuste c·baseⁿ
CENARIOS = ("melhor", "tipico", "pior")
TEMPO_MINIMO_NS = 2_000_000  # repete a medição até somar 2 ms (seleções rápidas são ruidosas)
TEMPO_UMA_REPETICAO_NS = 20_000_000  # uma seleção de 20 ms ou mais já é medida uma vez só
MAX_REPETICOES = 50


@dataclass(frozen=True)
class Grade:
    """Tamanhos medidos em cada varredura."""

    n: tuple[int, ...] = (5, 10, 20, 50, 100, 200, 400, 800)
    t: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)
    casos_memoria: int = 3
    n_forca_bruta: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20)
    t_forca_bruta: int = 512
    casos_forca_bruta: int = 20
    n_cenarios: tuple[int, ...] = (10, 50, 200, 800)
    t_cenarios: int = 1024
    casos_cenarios: int = 30
    fatores_escala: tuple[int, ...] = (1, 2, 4, 8, 16)
    n_escala: int = 50
    t_escala: int = 512
    casos_escala: int = 30

    def como_dict(self) -> dict:
        return {nome: list(valor) if isinstance(valor, tuple) else valor for nome, valor in self.__dict__.items()}


GRADE_RAPIDA = Grade(
    n=(5, 10, 20, 50),
    t=(128, 512, 2048),
    n_forca_bruta=(4, 6, 8, 10, 12, 14),
    casos_forca_bruta=3,
    n_cenarios=(10, 50),
    casos_cenarios=3,
    fatores_escala=(1, 2, 4),
    casos_escala=3,
)


# ---------------------------------------------------------------------------------------
# Força bruta (referência exponencial)


@dataclass(frozen=True)
class ForcaBruta:
    """Enumera os 2^n subconjuntos e fica com o de maior valor que cabe em T.

    Exata como a DP, mas Θ(2^n) em tempo e memória: cada subconjunto (máscara de bits)
    reaproveita o peso e o valor do subconjunto sem o seu bit mais baixo.
    """

    nome: ClassVar[str] = "forca_bruta"

    def selecionar(self, chunks: Sequence[Chunk], orcamento: int) -> Selecao:
        itens = [chunk for chunk in chunks if chunk.score > 0]
        pesos_itens = [chunk.tokens for chunk in itens]
        valores_itens = [chunk.score for chunk in itens]
        total = 1 << len(itens)
        pesos = array("q", bytes(8 * total))
        valores = array("d", bytes(8 * total))
        melhor_mascara, melhor_valor = 0, 0.0
        for mascara in range(1, total):
            bit = mascara & -mascara
            i = bit.bit_length() - 1
            peso = pesos[mascara ^ bit] + pesos_itens[i]
            valor = valores[mascara ^ bit] + valores_itens[i]
            pesos[mascara] = peso
            valores[mascara] = valor
            if peso <= orcamento and valor > melhor_valor:
                melhor_mascara, melhor_valor = mascara, valor
        escolhidos = [itens[i] for i in range(len(itens)) if melhor_mascara >> i & 1]
        posicao = {id(chunk): i for i, chunk in enumerate(chunks)}
        ordem = tuple(sorted(escolhidos, key=lambda chunk: (-chunk.score, posicao[id(chunk)])))
        return Selecao(self.nome, ordem, orcamento)


ALGORITMOS_ASSINTOTICOS = (*ALGORITMOS, "forca_bruta")


def criar_seletor_assintotico(algoritmo: str, k: int) -> Seletor:
    return ForcaBruta() if algoritmo == "forca_bruta" else criar_seletor(algoritmo, k=k)


# ---------------------------------------------------------------------------------------
# Réplicas instrumentadas: executam os mesmos passos do código e contam cada um


@dataclass
class Contagem:
    passos: int = 0  # passos de laço, cálculo de chaves, montagem de listas e dicionários
    comparacoes: int = 0  # comparações feitas pelas ordenações (Timsort do Python)
    celulas: int = 0  # células da DP, ou subconjuntos da força bruta
    melhorias: int = 0  # células que gravaram um valor melhor (já incluídas em `celulas`)
    inicializacao: int = 0  # posições de memória zeradas na alocação (vetor/tabela da DP, arrays)

    @property
    def total(self) -> int:
        return self.passos + self.comparacoes + self.celulas + self.inicializacao


def _ordenar(itens: Sequence, chave: Callable, contagem: Contagem, reverso: bool = False) -> list:
    """O mesmo `sorted(itens, key=chave, reverse=reverso)`, contando chaves e comparações."""

    class Chave:
        __slots__ = ("item", "valor")

        def __init__(self, valor, item) -> None:
            self.valor, self.item = valor, item

        def __lt__(self, outra: Chave) -> bool:
            contagem.comparacoes += 1
            return self.valor < outra.valor

    contagem.passos += len(itens)
    return [envolto.item for envolto in sorted([Chave(chave(item), item) for item in itens], reverse=reverso)]


def _ordem_do_prompt(escolhidos: Sequence[Chunk], candidatos: Sequence[Chunk], contagem: Contagem) -> list[Chunk]:
    """Réplica de seletores._na_ordem_do_prompt: dicionário de posições e ordenação por (-score, posição)."""
    contagem.passos += len(candidatos)
    posicao = {id(chunk): i for i, chunk in enumerate(candidatos)}
    return _ordenar(escolhidos, lambda chunk: (-chunk.score, posicao[id(chunk)]), contagem)


def _densidade(chunk: Chunk) -> float:
    return chunk.score / chunk.tokens if chunk.tokens else math.inf


def _topk_contando(chunks: Sequence[Chunk], orcamento: int, k: int | None, respeitar: bool, c: Contagem) -> list:
    ordenados = _ordem_do_prompt(chunks, chunks, c)
    limite = len(ordenados) if k is None else k
    if not respeitar:
        escolhidos = ordenados[:limite]
        c.passos += len(escolhidos)
        return escolhidos
    escolhidos, usados = [], 0
    for chunk in ordenados:
        c.passos += 1
        if len(escolhidos) == limite:
            break
        if usados + chunk.tokens <= orcamento:
            escolhidos.append(chunk)
            usados += chunk.tokens
    return escolhidos


def _greedy_contando(chunks: Sequence[Chunk], orcamento: int, c: Contagem) -> list:
    escolhidos, usados = [], 0
    for chunk in _ordenar(chunks, _densidade, c, reverso=True):
        c.passos += 1
        if chunk.score > 0 and usados + chunk.tokens <= orcamento:
            escolhidos.append(chunk)
            usados += chunk.tokens
    return _ordem_do_prompt(escolhidos, chunks, c)


def _knapsack_contando(chunks: Sequence[Chunk], orcamento: int, c: Contagem) -> list:
    c.passos += len(chunks)
    itens = [chunk for chunk in chunks if chunk.score > 0 and chunk.tokens <= orcamento]
    c.inicializacao += orcamento + 1
    melhor = [0.0] * (orcamento + 1)
    c.passos += len(itens)
    c.inicializacao += len(itens) * (orcamento + 1)
    pegou = [bytearray(orcamento + 1) for _ in itens]
    melhorias = 0
    for i, chunk in enumerate(itens):
        c.passos += 1
        peso, valor = chunk.tokens, chunk.score
        c.celulas += orcamento - peso + 1  # o laço abaixo roda exatamente esse número de vezes
        decisoes = pegou[i]
        for t in range(orcamento, peso - 1, -1):
            if melhor[t - peso] + valor > melhor[t]:
                melhor[t] = melhor[t - peso] + valor
                decisoes[t] = 1
                melhorias += 1
    c.melhorias += melhorias
    escolhidos, t = [], orcamento
    for i in range(len(itens) - 1, -1, -1):
        c.passos += 1
        if pegou[i][t]:
            escolhidos.append(itens[i])
            t -= itens[i].tokens
    return _ordem_do_prompt(escolhidos, chunks, c)


def _forca_bruta_contando(chunks: Sequence[Chunk], orcamento: int, c: Contagem) -> list:
    c.passos += len(chunks)
    itens = [chunk for chunk in chunks if chunk.score > 0]
    c.passos += 2 * len(itens)
    pesos_itens = [chunk.tokens for chunk in itens]
    valores_itens = [chunk.score for chunk in itens]
    total = 1 << len(itens)
    c.inicializacao += 2 * total
    pesos = array("q", bytes(8 * total))
    valores = array("d", bytes(8 * total))
    melhor_mascara, melhor_valor, melhorias = 0, 0.0, 0
    for mascara in range(1, total):
        bit = mascara & -mascara
        i = bit.bit_length() - 1
        peso = pesos[mascara ^ bit] + pesos_itens[i]
        valor = valores[mascara ^ bit] + valores_itens[i]
        pesos[mascara] = peso
        valores[mascara] = valor
        if peso <= orcamento and valor > melhor_valor:
            melhor_mascara, melhor_valor = mascara, valor
            melhorias += 1
    c.celulas += total - 1
    c.melhorias += melhorias
    c.passos += len(itens)
    escolhidos = [itens[i] for i in range(len(itens)) if melhor_mascara >> i & 1]
    return _ordem_do_prompt(escolhidos, chunks, c)


def executar_contando(algoritmo: str, chunks: Sequence[Chunk], orcamento: int, k: int) -> tuple[list[Chunk], Contagem]:
    """Roda a réplica instrumentada e devolve os chunks escolhidos (na ordem do prompt) e a contagem."""
    c = Contagem()
    if algoritmo == "topk":
        escolhidos = _topk_contando(chunks, orcamento, k, False, c)
    elif algoritmo == "topk_corte":
        escolhidos = _topk_contando(chunks, orcamento, None, True, c)
    elif algoritmo == "greedy":
        escolhidos = _greedy_contando(chunks, orcamento, c)
    elif algoritmo == "knapsack":
        escolhidos = _knapsack_contando(chunks, orcamento, c)
    elif algoritmo == "forca_bruta":
        escolhidos = _forca_bruta_contando(chunks, orcamento, c)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algoritmo!r}")
    return escolhidos, c


def celulas_da_dp(chunks: Sequence[Chunk], orcamento: int) -> int:
    """Fórmula teórica das células da DP: Σ (T - w_i + 1) sobre os itens com score > 0 e w_i <= T."""
    return sum(orcamento - c.tokens + 1 for c in chunks if c.score > 0 and c.tokens <= orcamento)


def expoente_efetivo_nlogn(ns: Sequence[float]) -> float:
    """Expoente que n·log n aparenta quando ajustado como n^a nesta faixa de n (> 1, tende a 1 devagar)."""
    ns = [n for n in ns if n >= 2]
    if len(set(ns)) < 2:
        return 1.0
    return ajuste_potencia(ns, [n * math.log2(n) for n in ns])["expoente"]


# ---------------------------------------------------------------------------------------
# Medição


def cronometrar(funcao: Callable[[], object]) -> tuple[int, int]:
    """Mediana do tempo de `funcao`, em ns, e o número de repetições medidas.

    O coletor de lixo fica desligado durante a medição (como no `timeit`). A primeira
    execução é aquecimento; se ela já levou 20 ms ou mais, vale como a medida.
    """
    gc_estava_ligado = gc.isenabled()
    gc.disable()
    try:
        inicio = time.perf_counter_ns()
        funcao()
        primeira = time.perf_counter_ns() - inicio
        if primeira >= TEMPO_UMA_REPETICAO_NS:
            return primeira, 1
        tempos: list[int] = []
        while not tempos or (sum(tempos) < TEMPO_MINIMO_NS and len(tempos) < MAX_REPETICOES):
            inicio = time.perf_counter_ns()
            funcao()
            tempos.append(time.perf_counter_ns() - inicio)
        return int(statistics.median(tempos)), len(tempos)
    finally:
        if gc_estava_ligado:
            gc.enable()


def pico_de_memoria_kib(funcao: Callable[[], object]) -> float:
    tracemalloc.start()
    try:
        funcao()
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pico / 1024


def extras_do_caso(pool: Sequence[Exemplo], caso: int, semente: int) -> list[Chunk]:
    """Parágrafos de outras perguntas, numa ordem fixa por caso (as instâncias ficam aninhadas)."""
    outros = [chunk for i, exemplo in enumerate(pool) if i != caso for chunk in exemplo.chunks]
    random.Random(f"{semente}:{caso}").shuffle(outros)
    return outros


def montar_instancia(exemplo: Exemplo, extras: Sequence[Chunk], n: int) -> list[Chunk]:
    proprios = list(exemplo.chunks)
    if n <= len(proprios):
        return proprios[:n]
    faltam = n - len(proprios)
    if faltam > len(extras):
        raise ValueError(f"Não há parágrafos suficientes para n = {n}; use mais perguntas no pool")
    return proprios + [
        replace(chunk, id=f"extra{j}:{chunk.id}", de_suporte=False) for j, chunk in enumerate(extras[:faltam])
    ]


def instancia_do_cenario(
    chunks: Sequence[Chunk], algoritmo: str, cenario: str, orcamento: int, semente: str
) -> list[Chunk]:
    """Transforma a instância típica no melhor ou no pior caso de cada algoritmo.

    Os que ordenam: melhor = entrada já na ordem da chave (o Timsort faz n - 1 comparações);
    pior = entrada embaralhada (o Timsort precisa de Θ(n log n) comparações).
    Knapsack: melhor = nenhum chunk cabe (custo Θ(n + T)); pior = todos cabem com folga
    (pesos ÷ 16), e as células chegam a ≈ n·T.
    """
    if cenario == "tipico":
        return list(chunks)
    if algoritmo == "knapsack":
        if cenario == "melhor":
            return [replace(c, tokens=orcamento + c.tokens) for c in chunks]
        return [replace(c, tokens=max(1, c.tokens // 16)) for c in chunks]
    if cenario == "melhor":
        chave = _densidade if algoritmo == "greedy" else (lambda c: c.score)
        return sorted(chunks, key=chave, reverse=True)
    embaralhados = list(chunks)
    random.Random(semente).shuffle(embaralhados)
    return embaralhados


def _maior_valor_que_cabe(chunks: Sequence[Chunk], orcamento: int) -> float:
    return max((c.score for c in chunks if c.score > 0 and c.tokens <= orcamento), default=0.0)


def medir_configuracao(
    algoritmos: Sequence[str], chunks: list[Chunk], orcamento: int, k: int, base: dict, otimo: float | None = None
) -> list[dict]:
    """Mede os algoritmos numa instância (n, T), confere a réplica e compara cada um com o ótimo."""
    if otimo is None:
        otimo = Knapsack().selecionar(chunks, orcamento).valor_total
    v_max = _maior_valor_que_cabe(chunks, orcamento)
    linhas = []
    for algoritmo in algoritmos:
        seletor = criar_seletor_assintotico(algoritmo, k)
        selecao = seletor.selecionar(chunks, orcamento)
        replica, contagem = executar_contando(algoritmo, chunks, orcamento, k)
        if [id(c) for c in replica] != [id(c) for c in selecao.chunks]:
            raise AssertionError(
                f"A réplica instrumentada de {algoritmo} divergiu do código (n = {len(chunks)}, T = {orcamento})"
            )
        tempo_ns, repeticoes = cronometrar(lambda s=seletor, c=chunks, t=orcamento: s.selecionar(c, t))
        linhas.append(
            {
                **base,
                "algoritmo": algoritmo,
                "n": len(chunks),
                "T": orcamento,
                "tempo_ns": tempo_ns,
                "repeticoes": repeticoes,
                "operacoes": contagem.total,
                "passos": contagem.passos,
                "comparacoes": contagem.comparacoes,
                "celulas": contagem.celulas,
                "melhorias": contagem.melhorias,
                "inicializacao": contagem.inicializacao,
                "valor": selecao.valor_total,
                "valor_otimo": otimo,
                "razao_otimo": selecao.valor_total / otimo if otimo > 0 else 1.0,
                "perda": otimo - selecao.valor_total,
                "v_max": v_max,
                "estourou": selecao.estourou_orcamento,
                "tokens_usados": selecao.tokens_usados,
                "escolhidos": len(selecao.chunks),
            }
        )
    return linhas


def _medir_memoria(algoritmos: Sequence[str], chunks: list[Chunk], orcamento: int, k: int, base: dict) -> list[dict]:
    linhas = []
    for algoritmo in algoritmos:
        seletor = criar_seletor_assintotico(algoritmo, k)
        kib = pico_de_memoria_kib(lambda s=seletor, c=chunks, t=orcamento: s.selecionar(c, t))
        linhas.append({**base, "algoritmo": algoritmo, "n": len(chunks), "T": orcamento, "memoria_kib": kib})
    return linhas


def executar_varreduras(
    casos: Sequence[Exemplo],
    pool: Sequence[Exemplo],
    grade: Grade,
    k: int,
    semente: int = 2026,
    progresso: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Roda as quatro varreduras; devolve as medidas e as medições de memória."""
    medidas: list[dict] = []
    memoria: list[dict] = []

    def avisar(texto: str) -> None:
        if progresso:
            progresso(texto)

    for caso, exemplo in enumerate(casos):
        extras = extras_do_caso(pool, caso, semente)
        base = {"varredura": "principal", "caso": caso, "pergunta_id": exemplo.id}
        for n in grade.n:
            chunks = montar_instancia(exemplo, extras, n)
            for orcamento in grade.t:
                medidas.extend(medir_configuracao(ALGORITMOS, chunks, orcamento, k, base))
                if caso < grade.casos_memoria:
                    memoria.extend(_medir_memoria(ALGORITMOS, chunks, orcamento, k, base))
        avisar(f"varredura principal: caso {caso + 1}/{len(casos)}")

    for caso, exemplo in enumerate(casos[: grade.casos_forca_bruta]):
        extras = extras_do_caso(pool, caso, semente)
        base = {"varredura": "forca_bruta", "caso": caso, "pergunta_id": exemplo.id}
        for n in grade.n_forca_bruta:
            chunks = montar_instancia(exemplo, extras, n)
            bruta, dp = medir_configuracao(("forca_bruta", "knapsack"), chunks, grade.t_forca_bruta, k, base)
            if abs(bruta["valor"] - dp["valor"]) > 1e-9:
                raise AssertionError(f"Força bruta e DP divergem no caso {caso}, n = {n}")
            medidas.extend([bruta, dp])
            if caso < grade.casos_memoria:
                memoria.extend(_medir_memoria(("forca_bruta", "knapsack"), chunks, grade.t_forca_bruta, k, base))
        avisar(f"força bruta: caso {caso + 1}/{min(grade.casos_forca_bruta, len(casos))}")

    for caso, exemplo in enumerate(casos[: grade.casos_cenarios]):
        extras = extras_do_caso(pool, caso, semente)
        for n in grade.n_cenarios:
            tipica = montar_instancia(exemplo, extras, n)
            for algoritmo in ALGORITMOS:
                for cenario in CENARIOS:
                    chunks = instancia_do_cenario(tipica, algoritmo, cenario, grade.t_cenarios, f"{semente}:{caso}:{n}")
                    base = {"varredura": "cenarios", "caso": caso, "pergunta_id": exemplo.id, "cenario": cenario}
                    medidas.extend(medir_configuracao((algoritmo,), chunks, grade.t_cenarios, k, base))
        avisar(f"melhor/pior caso: caso {caso + 1}/{min(grade.casos_cenarios, len(casos))}")

    for caso, exemplo in enumerate(casos[: grade.casos_escala]):
        extras = extras_do_caso(pool, caso, semente)
        original = montar_instancia(exemplo, extras, grade.n_escala)
        otimo = None
        for fator in grade.fatores_escala:
            chunks = [replace(chunk, tokens=chunk.tokens * fator) for chunk in original]
            orcamento = grade.t_escala * fator
            base = {"varredura": "escala", "caso": caso, "pergunta_id": exemplo.id, "fator": fator}
            linhas = medir_configuracao(ALGORITMOS, chunks, orcamento, k, base)
            valor_dp = next(linha["valor"] for linha in linhas if linha["algoritmo"] == "knapsack")
            if otimo is None:
                otimo = valor_dp
            elif abs(valor_dp - otimo) > 1e-9:
                raise AssertionError(f"Escalar pesos e T mudou o ótimo no caso {caso} (fator {fator})")
            medidas.extend(linhas)
        avisar(f"escala dos pesos: caso {caso + 1}/{min(grade.casos_escala, len(casos))}")
    return medidas, memoria


# ---------------------------------------------------------------------------------------
# Ajustes (expoentes empíricos)


def ajuste_potencia(x: Sequence[float], y: Sequence[float]) -> dict:
    """Ajusta y ≈ c·x^a por mínimos quadrados em log-log; devolve a, c, R² e o erro-padrão de a."""
    lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))
    if len(lx) < 2 or np.ptp(lx) == 0:
        return {"expoente": float("nan"), "coeficiente": float("nan"), "r2": float("nan"), "erro_padrao": float("nan")}
    a, log_c = np.polyfit(lx, ly, 1)
    previsto = a * lx + log_c
    residuo = float(np.sum((ly - previsto) ** 2))
    total = float(np.sum((ly - ly.mean()) ** 2))
    erro = math.sqrt(residuo / (len(lx) - 2) / float(np.sum((lx - lx.mean()) ** 2))) if len(lx) > 2 else float("nan")
    return {
        "expoente": float(a),
        "coeficiente": float(math.exp(log_c)),
        "r2": 1 - residuo / total if total > 0 else 1.0,
        "erro_padrao": erro,
    }


def ajuste_n_e_t(n: Sequence[float], t: Sequence[float], y: Sequence[float]) -> dict:
    """Ajusta y ≈ c·n^a·T^b (regressão linear em log); a e b são os expoentes em n e em T."""
    ln, lt, ly = (np.log(np.asarray(v, dtype=float)) for v in (n, t, y))
    matriz = np.column_stack([ln, lt, np.ones_like(ln)])
    coeficientes, *_ = np.linalg.lstsq(matriz, ly, rcond=None)
    a, b, log_c = (float(v) for v in coeficientes)
    residuo = ly - matriz @ coeficientes
    total = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - float(np.sum(residuo**2)) / total if total > 0 else 1.0
    erros = [float("nan"), float("nan")]
    graus = len(ly) - 3
    if graus > 0:
        covariancia = float(np.sum(residuo**2)) / graus * np.linalg.pinv(matriz.T @ matriz)
        erros = [math.sqrt(max(covariancia[0, 0], 0.0)), math.sqrt(max(covariancia[1, 1], 0.0))]
    return {
        "expoente_n": a,
        "expoente_T": b,
        "erro_padrao_n": erros[0],
        "erro_padrao_T": erros[1],
        "coeficiente": math.exp(log_c),
        "r2": r2,
    }


def ajuste_exponencial(n: Sequence[float], y: Sequence[float]) -> dict:
    """Ajusta y ≈ c·base^n (log-linear); para a força bruta espera-se base ≈ 2."""
    nn, ly = np.asarray(n, dtype=float), np.log(np.asarray(y, dtype=float))
    inclinacao, log_c = np.polyfit(nn, ly, 1)
    previsto = inclinacao * nn + log_c
    total = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - float(np.sum((ly - previsto) ** 2)) / total if total > 0 else 1.0
    return {"base": float(math.exp(inclinacao)), "coeficiente": float(math.exp(log_c)), "r2": r2}


def medianas(linhas: Sequence[dict], campo: str) -> dict[tuple[str, int, int], float]:
    """Mediana de `campo` sobre os casos, por (algoritmo, n, T)."""
    grupos: dict[tuple[str, int, int], list[float]] = {}
    for linha in linhas:
        grupos.setdefault((linha["algoritmo"], linha["n"], linha["T"]), []).append(linha[campo])
    return {chave: float(statistics.median(valores)) for chave, valores in grupos.items()}


def resumo_dos_ajustes(
    medidas: Sequence[dict],
    n_minimo: int = REGIME_N,
    t_minimo: int = REGIME_T,
    n_minimo_forca_bruta: int = N_MINIMO_FORCA_BRUTA,
) -> list[dict]:
    """Expoentes de tempo e de operações por algoritmo, na grade inteira e nas entradas grandes."""
    principal = [linha for linha in medidas if linha["varredura"] == "principal"]
    resumo = []
    for regime, (n_min, t_min) in (("grade inteira", (0, 0)), ("entradas grandes", (n_minimo, t_minimo))):
        for campo in ("tempo_ns", "operacoes"):
            mediana = medianas([linha for linha in principal if linha["n"] >= n_min and linha["T"] >= t_min], campo)
            for algoritmo in ALGORITMOS:
                pontos = [(n, t, y) for (a, n, t), y in mediana.items() if a == algoritmo]
                if len({p[0] for p in pontos}) < 2 or len({p[1] for p in pontos}) < 2:
                    continue  # com um só n (ou um só T) o expoente daquele eixo não é identificável
                n, t, y = zip(*pontos, strict=True)
                resumo.append({"regime": regime, "medida": campo, "algoritmo": algoritmo, **ajuste_n_e_t(n, t, y)})
    forca = [linha for linha in medidas if linha["varredura"] == "forca_bruta" and linha["algoritmo"] == "forca_bruta"]
    for campo in ("tempo_ns", "operacoes"):
        mediana = medianas(forca, campo)
        grandes = {chave: y for chave, y in mediana.items() if chave[1] >= n_minimo_forca_bruta}
        pontos = grandes if len(grandes) >= 3 else mediana
        if len(pontos) >= 2:
            ajuste = ajuste_exponencial([chave[1] for chave in pontos], list(pontos.values()))
            resumo.append(
                {"regime": f"n >= {min(c[1] for c in pontos)}", "medida": campo, "algoritmo": "forca_bruta", **ajuste}
            )
    return resumo


# ---------------------------------------------------------------------------------------
# Ambiente de medição


def fixar_nucleo() -> dict:
    """Fixa o processo num núcleo de desempenho (CPUs híbridas variam ~2× entre núcleos)."""
    if not hasattr(os, "sched_setaffinity"):
        return {"afinidade": "não suportada neste sistema"}
    disponiveis = sorted(os.sched_getaffinity(0))
    desempenho: list[int] = []
    try:
        texto = Path("/sys/devices/cpu_core/cpus").read_text().strip()
        for parte in texto.split(","):
            inicio, _, fim = parte.partition("-")
            desempenho.extend(range(int(inicio), int(fim or inicio) + 1))
    except (OSError, ValueError):
        pass
    candidatos = [cpu for cpu in disponiveis if cpu in desempenho] or disponiveis
    escolhido = candidatos[-1]
    os.sched_setaffinity(0, {escolhido})
    governador = "?"
    try:
        governador = Path(f"/sys/devices/system/cpu/cpu{escolhido}/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    tipo = "desempenho (P-core)" if escolhido in desempenho else "único núcleo disponível"
    return {"afinidade": f"CPU {escolhido}, {tipo}", "governador": governador}


def maquina() -> dict:
    processador = platform.processor()
    try:
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name"):
                processador = linha.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "processador": processador,
        "nucleos": os.cpu_count(),
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
    }


def gravar_analise(pasta: Path, medidas: Sequence[dict], memoria: Sequence[dict], execucao: dict) -> None:
    """Grava a pasta que o painel assintótico lê."""
    gravar_jsonl(medidas, pasta / "medidas.jsonl")
    gravar_jsonl(memoria, pasta / "memoria.jsonl")
    (pasta / "execucao.json").write_text(json.dumps(execucao, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--casos", type=inteiro_positivo, default=200, help="perguntas usadas como casos (padrão: 200)")
    parser.add_argument("--n", type=inteiro_positivo, nargs="+", default=None, help="tamanhos de instância")
    parser.add_argument("--orcamentos", type=inteiro_positivo, nargs="+", default=None, help="valores de T")
    parser.add_argument("--rapido", action="store_true", help="grade pequena, para conferir rápido")
    parser.add_argument("--fonte", choices=("auto", "dev", "amostra"), default="auto")
    parser.add_argument("--k", type=inteiro_positivo, default=None, help="k do Top-K (padrão: TOP_K do .env)")
    parser.add_argument("--semente", type=int, default=2026)
    parser.add_argument("--sem-afinidade", action="store_true", help="não fixa o processo num núcleo")
    args = parser.parse_args(argv)

    cfg = Config.do_ambiente()
    if cfg.precisa_chave_openai and not tem_chave_openai():
        print("Os embeddings da OpenAI exigem OPENAI_API_KEY; ou rode com EMBEDDER=fastembed.", file=sys.stderr)
        return 1
    grade = GRADE_RAPIDA if args.rapido else Grade()
    if args.n:
        grade = replace(grade, n=tuple(sorted(set(args.n))))
    if args.orcamentos:
        grade = replace(grade, t=tuple(sorted(set(args.orcamentos))))
    k = args.k or cfg.top_k

    dev = caminho_dev(cfg)
    if args.fonte == "dev" and not dev.exists():
        sys.exit("Dataset não encontrado. Baixe com: uv run task dados")
    fonte = CAMINHO_AMOSTRA if args.fonte == "amostra" or (args.fonte == "auto" and not dev.exists()) else dev
    # O pool precisa de parágrafos suficientes para a maior instância (cerca de 10 por pergunta).
    maior_n = max([*grade.n, *grade.n_forca_bruta, *grade.n_cenarios, grade.n_escala])
    exemplos = carregar_exemplos(fonte, max(args.casos, math.ceil(maior_n / 10) + 2))
    casos = exemplos[: args.casos]
    print(f"{len(exemplos)} perguntas de {fonte.name} (pool) | {len(casos)} casos")
    print(f"n = {list(grade.n)} | T = {list(grade.t)}")
    recuperador = criar_recuperador(cfg)
    exemplos = [recuperador.preparar(exemplo) for exemplo in exemplos]
    recuperador.salvar_cache()
    casos = exemplos[: args.casos]
    ambiente = {"afinidade": "livre (--sem-afinidade)"} if args.sem_afinidade else fixar_nucleo()
    print(f"Medindo em {ambiente['afinidade']}, coletor de lixo desligado durante as medições")

    inicio = time.perf_counter()
    medidas, memoria = executar_varreduras(
        casos,
        exemplos,
        grade,
        k,
        args.semente,
        progresso=lambda texto: print(f"\r{texto:<60}", end="", file=sys.stderr, flush=True),
    )
    print(file=sys.stderr)
    duracao = time.perf_counter() - inicio
    ajustes = resumo_dos_ajustes(medidas)

    agora = datetime.now()
    pasta = cfg.dir_resultados / f"assintotica-{agora:%Y%m%d-%H%M%S}"
    execucao = {
        "data": agora.isoformat(timespec="seconds"),
        "fonte": fonte.name,
        "casos": len(casos),
        "pool": len(exemplos),
        "n": list(grade.n),
        "T": list(grade.t),
        "k": k,
        "grade": grade.como_dict(),
        "regime": {"n_minimo": REGIME_N, "t_minimo": REGIME_T, "n_minimo_forca_bruta": N_MINIMO_FORCA_BRUTA},
        "semente": args.semente,
        "embedder": f"{cfg.embedder}/{cfg.modelo_embedding}",
        "tokenizador": recuperador.contador.nome,
        "maquina": {**maquina(), **ambiente, "coletor_de_lixo": "desligado durante as medições"},
        "duracao_s": duracao,
        "ajustes": ajustes,
    }
    gravar_analise(pasta, medidas, memoria, execucao)

    print(f"{len(medidas)} medidas em {duracao:.0f} s (réplicas conferidas em todas)")
    for ajuste in ajustes:
        if ajuste["regime"] == "grade inteira":
            continue
        if "expoente_n" in ajuste:
            print(
                f"  [{ajuste['regime']}] {ajuste['medida']:>9}  {ajuste['algoritmo']:<11} ~ "
                f"n^{ajuste['expoente_n']:.2f} · T^{ajuste['expoente_T']:.2f}  (R² = {ajuste['r2']:.3f})"
            )
        else:
            rotulo = f"[{ajuste['regime']}] {ajuste['medida']:>9}  {ajuste['algoritmo']:<11}"
            print(f"  {rotulo} ~ {ajuste['base']:.3f}^n  (R² = {ajuste['r2']:.3f})")
    print(f"Resultados em {pasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
