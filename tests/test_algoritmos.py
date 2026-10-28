"""Os três testes do HotpotQA para cada algoritmo, um por etapa do pipeline.

1. Seleção (sem LLM): o contexto escolhido respeita o algoritmo e traz os parágrafos de suporte?
2. Geração com as métricas oficiais do HotpotQA (EM e F1): a resposta bate com o gabarito?
3. Agent-as-a-judge (AgentAsJudgeEval do agno): a resposta é equivalente ao gabarito,
   mesmo escrita de outro jeito?

Os limites mínimos são barras de sanidade, não metas de qualidade. Os números de cada
teste aparecem com `uv run task test -s`.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from rag_knapsack.config import Config
from rag_knapsack.evals import formatar_resumo
from rag_knapsack.metricas import exact_match
from rag_knapsack.pipeline import Resultado, resumir
from rag_knapsack.seletores import ALGORITMOS

RECALL_SUPORTE_MINIMO = 0.5  # sortear 4 dos 10 parágrafos daria recall esperado de 0,4
F1_MINIMO = 0.4
ACURACIA_JUIZ_MINIMA = 0.5


def _otimo_por_forca_bruta(resultado: Resultado) -> float:
    chunks, orcamento = resultado.exemplo.chunks, resultado.selecao.orcamento
    return max(
        sum(c.score for c in combinacao if c.score > 0)
        for tamanho in range(len(chunks) + 1)
        for combinacao in combinations(chunks, tamanho)
        if sum(c.tokens for c in combinacao) <= orcamento
    )


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_selecao_traz_paragrafos_de_suporte(cfg: Config, selecoes: dict[str, list[Resultado]], algoritmo: str) -> None:
    resultados = selecoes[algoritmo]
    for resultado in resultados:
        selecao = resultado.selecao
        if algoritmo == "topk":
            maiores = sorted((c.score for c in resultado.exemplo.chunks), reverse=True)[: cfg.top_k]
            assert [c.score for c in selecao.chunks] == maiores, resultado.exemplo.id
        else:
            assert not selecao.estourou_orcamento, resultado.exemplo.id
        if algoritmo == "knapsack":  # 10 candidatos: dá para conferir todos os 1.024 subconjuntos
            assert selecao.valor_total == pytest.approx(_otimo_por_forca_bruta(resultado)), resultado.exemplo.id

    resumo = resumir(resultados, cfg)
    print(f"\n{algoritmo} (k={cfg.top_k}, T={cfg.orcamento_tokens}), seleção:\n{formatar_resumo(resumo)}")
    assert resumo["recall_suporte"] >= RECALL_SUPORTE_MINIMO


@pytest.mark.llm
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_resposta_bate_com_gabarito_em_f1(cfg: Config, respostas: dict[str, list[Resultado]], algoritmo: str) -> None:
    resultados = respostas[algoritmo]
    erros = [resultado.erro for resultado in resultados if resultado.erro]
    assert not erros, erros

    resumo = resumir(resultados, cfg)
    print(f"\n{algoritmo} com {cfg.modelo_gerador}, EM/F1:\n{formatar_resumo(resumo)}")
    assert resumo["f1"] >= F1_MINIMO


@pytest.mark.llm
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_agent_as_judge(cfg: Config, respostas: dict[str, list[Resultado]], algoritmo: str) -> None:
    resultados = respostas[algoritmo]
    sem_veredito = [resultado.exemplo.id for resultado in resultados if resultado.veredito is None]
    assert not sem_veredito, [resultado.erro for resultado in resultados if resultado.erro]

    # Sanidade do próprio juiz: uma resposta idêntica ao gabarito não pode ser reprovada.
    reprovou_identica = [
        resultado.exemplo.id
        for resultado in resultados
        if exact_match(resultado.resposta.texto, resultado.exemplo.resposta) and not resultado.veredito.correto
    ]
    assert not reprovou_identica

    resumo = resumir(resultados, cfg)
    print(f"\n{algoritmo} julgado por {cfg.modelo_juiz}:\n{formatar_resumo(resumo)}")
    for resultado in resultados:
        marca = "ok " if resultado.veredito.correto else "ERR"
        resposta, gabarito = resultado.resposta.texto, resultado.exemplo.resposta
        print(f"  [{marca}] {resposta!r} vs {gabarito!r}: {resultado.veredito.justificativa}")
    assert resumo["acuracia_juiz"] >= ACURACIA_JUIZ_MINIMA
