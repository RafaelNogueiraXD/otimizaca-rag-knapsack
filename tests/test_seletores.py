"""Testes unitários dos seletores (sem rede): exemplo da proposta, casos de borda e força bruta."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from rag_knapsack.dados import Chunk
from rag_knapsack.seletores import ALGORITMOS, Greedy, Knapsack, TopK, criar_seletor

# Exemplo da seção 8 da proposta: "Onde fica Brasília?", T = 300.
BRASILIA = [
    ("c1", 280, 0.93),
    ("c2", 160, 0.88),
    ("c3", 140, 0.80),
    ("c4", 110, 0.70),
    ("c5", 90, 0.41),
    ("c6", 60, 0.22),
]


def chunks(especificacao: list[tuple[str, int, float]]) -> list[Chunk]:
    return [Chunk(id=i, titulo=i, texto="", tokens=w, score=v) for i, w, v in especificacao]


def ids(selecao) -> set[str]:
    return {chunk.id for chunk in selecao.chunks}


def otimo_por_forca_bruta(candidatos: list[Chunk], orcamento: int) -> float:
    melhor = 0.0
    for tamanho in range(len(candidatos) + 1):
        for combinacao in combinations(candidatos, tamanho):
            if sum(c.tokens for c in combinacao) <= orcamento:
                melhor = max(melhor, sum(c.score for c in combinacao if c.score > 0))
    return melhor


def test_exemplo_de_brasilia_da_proposta() -> None:
    candidatos = chunks(BRASILIA)
    topk = TopK(k=1).selecionar(candidatos, 300)
    greedy = Greedy().selecionar(candidatos, 300)
    knapsack = Knapsack().selecionar(candidatos, 300)

    assert ids(topk) == {"c1"} and topk.valor_total == pytest.approx(0.93)
    assert ids(greedy) == {"c3", "c4"} and greedy.valor_total == pytest.approx(1.50) and greedy.tokens_usados == 250
    assert ids(knapsack) == {"c2", "c3"} and knapsack.valor_total == pytest.approx(1.68)
    assert knapsack.tokens_usados == 300 and knapsack.ocupacao == pytest.approx(1.0)


def test_greedy_pode_perder_ate_do_top_k_com_corte() -> None:
    # T = 150: o guloso pega c4 (maior densidade) e depois nada mais cabe; o ótimo é c3.
    candidatos = chunks(BRASILIA)
    assert Greedy().selecionar(candidatos, 150).valor_total == pytest.approx(0.70)
    assert TopK(k=4, respeitar_orcamento=True).selecionar(candidatos, 150).valor_total == pytest.approx(0.80)
    assert Knapsack().selecionar(candidatos, 150).valor_total == pytest.approx(0.80)


def test_top_k_padrao_estoura_e_a_variante_com_corte_nao() -> None:
    candidatos = chunks(BRASILIA)
    padrao = TopK(k=2).selecionar(candidatos, 300)
    corte = TopK(k=2, respeitar_orcamento=True).selecionar(candidatos, 300)
    assert ids(padrao) == {"c1", "c2"} and padrao.estourou_orcamento
    assert not corte.estourou_orcamento and ids(corte) == {"c1"}
    assert padrao.algoritmo == "topk" and corte.algoritmo == "topk_corte"


@pytest.mark.parametrize("algoritmo", ["greedy", "knapsack", "topk_corte"])
def test_casos_de_borda(algoritmo: str) -> None:
    seletor = criar_seletor(algoritmo, k=4)
    assert seletor.selecionar([], 100).chunks == ()
    assert seletor.selecionar(chunks(BRASILIA), 50).chunks == ()  # nenhum chunk cabe
    with pytest.raises(ValueError):
        seletor.selecionar(chunks(BRASILIA), 0)


def test_greedy_pula_o_que_nao_cabe_e_continua() -> None:
    # Densidades: A 9,0 > B 3,3 > C 2,0 (×1000). B não cabe depois de A, mas C ainda cabe.
    candidatos = chunks([("A", 100, 0.9), ("B", 300, 1.0), ("C", 150, 0.3)])
    assert ids(Greedy().selecionar(candidatos, 260)) == {"A", "C"}


def test_top_k_com_corte_enche_o_orcamento_sem_limite_de_k() -> None:
    candidatos = chunks(BRASILIA)
    assert len(criar_seletor("topk_corte").selecionar(candidatos, 840).chunks) == 6
    assert len(TopK(k=4, respeitar_orcamento=True).selecionar(candidatos, 840).chunks) == 4  # para em k


def test_empate_de_score_da_a_mesma_ordem_em_todos_os_algoritmos() -> None:
    candidatos = chunks([("A", 120, 0.5), ("B", 80, 0.5), ("C", 100, 0.5)])
    ordens = {tuple(c.id for c in criar_seletor(a, k=3).selecionar(candidatos, 1000).chunks) for a in ALGORITMOS}
    assert ordens == {("A", "B", "C")}


def test_scores_nao_positivos_nunca_entram_na_mochila() -> None:
    candidatos = chunks([("bom", 50, 0.6), ("zero", 10, 0.0), ("negativo", 10, -0.3)])
    assert ids(Knapsack().selecionar(candidatos, 1000)) == {"bom"}
    assert ids(Greedy().selecionar(candidatos, 1000)) == {"bom"}


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_chunks_saem_na_ordem_do_prompt(algoritmo: str) -> None:
    selecao = criar_seletor(algoritmo, k=6).selecionar(chunks(BRASILIA), 840)
    scores = [chunk.score for chunk in selecao.chunks]
    assert scores == sorted(scores, reverse=True)
    assert len(ids(selecao)) == len(selecao.chunks)


def test_knapsack_bate_com_forca_bruta_e_domina_as_heuristicas() -> None:
    gerador = random.Random(2026)
    for _ in range(400):
        candidatos = [
            Chunk(
                id=str(i), titulo="", texto="", tokens=gerador.randint(1, 300), score=round(gerador.uniform(-0.1, 1), 3)
            )
            for i in range(gerador.randint(0, 10))
        ]
        orcamento = gerador.randint(1, 700)
        knapsack = Knapsack().selecionar(candidatos, orcamento)
        assert knapsack.tokens_usados <= orcamento
        assert knapsack.valor_total == pytest.approx(otimo_por_forca_bruta(candidatos, orcamento))
        for heuristica in (Greedy(), TopK(k=4, respeitar_orcamento=True)):
            selecao = heuristica.selecionar(candidatos, orcamento)
            assert selecao.tokens_usados <= orcamento
            assert selecao.valor_total <= knapsack.valor_total + 1e-9


def test_perda_do_greedy_limitada_por_um_chunk() -> None:
    """OPT - guloso <= valor do primeiro chunk (em densidade) que não coube."""
    gerador = random.Random(7)
    for _ in range(300):
        candidatos = [
            Chunk(
                id=str(i),
                titulo="",
                texto="",
                tokens=gerador.randint(20, 300),
                score=round(gerador.uniform(0.01, 1), 3),
            )
            for i in range(10)
        ]
        orcamento = gerador.randint(100, 900)
        perda = (
            Knapsack().selecionar(candidatos, orcamento).valor_total
            - Greedy().selecionar(candidatos, orcamento).valor_total
        )
        assert perda <= max(c.score for c in candidatos) + 1e-9


def test_criar_seletor_rejeita_nome_desconhecido() -> None:
    with pytest.raises(ValueError):
        criar_seletor("mochila-quantica")
