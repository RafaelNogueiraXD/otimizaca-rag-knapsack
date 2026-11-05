"""Testes da análise assintótica (sem rede): réplicas instrumentadas, cenários, instâncias e ajustes."""

from __future__ import annotations

import gc
import random
from pathlib import Path

import pytest

from rag_knapsack.assintotica import (
    ALGORITMOS_ASSINTOTICOS,
    ForcaBruta,
    Grade,
    ajuste_exponencial,
    ajuste_n_e_t,
    ajuste_potencia,
    celulas_da_dp,
    criar_seletor_assintotico,
    cronometrar,
    executar_contando,
    executar_varreduras,
    expoente_efetivo_nlogn,
    extras_do_caso,
    gravar_analise,
    instancia_do_cenario,
    montar_instancia,
    resumo_dos_ajustes,
)
from rag_knapsack.dados import Chunk
from rag_knapsack.seletores import ALGORITMOS, Knapsack

GRADE_TESTE = Grade(
    n=(5, 20),
    t=(128, 512),
    casos_memoria=1,
    n_forca_bruta=(4, 6),
    casos_forca_bruta=1,
    n_cenarios=(10, 20),
    casos_cenarios=1,
    fatores_escala=(1, 2),
    casos_escala=1,
)


def instancia(n: int, semente: int, scores_negativos: bool = False) -> list[Chunk]:
    gerador = random.Random(semente)
    minimo = -0.2 if scores_negativos else 0.05
    return [
        Chunk(
            id=str(i), titulo="", texto="", tokens=gerador.randint(20, 300), score=round(gerador.uniform(minimo, 1), 3)
        )
        for i in range(n)
    ]


def test_forca_bruta_chega_ao_mesmo_otimo_que_a_dp() -> None:
    for semente in range(60):
        chunks = instancia(random.Random(semente).randint(0, 12), semente)
        orcamento = random.Random(semente + 1).randint(1, 900)
        bruta = ForcaBruta().selecionar(chunks, orcamento)
        assert bruta.tokens_usados <= orcamento
        assert bruta.valor_total == pytest.approx(Knapsack().selecionar(chunks, orcamento).valor_total)


def test_replicas_escolhem_exatamente_o_mesmo_que_o_codigo() -> None:
    gerador = random.Random(11)
    for _ in range(250):
        chunks = instancia(gerador.randint(0, 13), gerador.randint(0, 10**6), scores_negativos=True)
        if chunks and gerador.random() < 0.3:  # empates de score
            chunks = [Chunk(id=c.id, titulo="", texto="", tokens=c.tokens, score=0.5) for c in chunks]
        orcamento = gerador.randint(1, 900)
        for algoritmo in ALGORITMOS_ASSINTOTICOS:
            real = criar_seletor_assintotico(algoritmo, 4).selecionar(chunks, orcamento)
            replica, contagem = executar_contando(algoritmo, chunks, orcamento, 4)
            assert [id(c) for c in replica] == [id(c) for c in real.chunks], algoritmo
            if algoritmo == "knapsack":
                assert contagem.celulas == celulas_da_dp(chunks, orcamento)
                assert contagem.melhorias <= contagem.celulas


def test_contagem_segue_a_complexidade() -> None:
    chunks = instancia(800, 3)

    def operacoes(algoritmo: str, n: int, orcamento: int) -> int:
        return executar_contando(algoritmo, chunks[:n], orcamento, 4)[1].total

    for algoritmo in ("topk", "topk_corte"):
        razao = operacoes(algoritmo, 800, 512) / operacoes(algoritmo, 400, 512)
        assert 2.0 < razao < 2.5, algoritmo  # dobrar n multiplica n log n por pouco mais de 2
        assert operacoes(algoritmo, 400, 8192) == operacoes(algoritmo, 400, 512)  # não depende de T
    assert 2.0 < operacoes("greedy", 800, 512) / operacoes("greedy", 400, 512) < 2.5
    # Knapsack com T muito maior que os pesos: dobrar T ou n quase dobra as operações.
    assert 1.9 < operacoes("knapsack", 100, 16384) / operacoes("knapsack", 100, 8192) < 2.1
    assert 1.9 < operacoes("knapsack", 200, 8192) / operacoes("knapsack", 100, 8192) < 2.1
    assert executar_contando("forca_bruta", chunks[:12], 512, 4)[1].celulas == 2**12 - 1


def test_melhor_e_pior_caso() -> None:
    chunks = instancia(200, 5)
    # Entrada já ordenada por score: o Timsort do Top-K faz n - 1 comparações.
    ordenada = instancia_do_cenario(chunks, "topk", "melhor", 1024, "x")
    assert executar_contando("topk", ordenada, 1024, 4)[1].comparacoes == 199
    embaralhada = instancia_do_cenario(chunks, "topk", "pior", 1024, "x")
    assert executar_contando("topk", embaralhada, 1024, 4)[1].comparacoes > 1000
    # Knapsack: no melhor caso nenhum chunk cabe; no pior, as células chegam perto de n·T.
    assert (
        executar_contando("knapsack", instancia_do_cenario(chunks, "knapsack", "melhor", 1024, "x"), 1024, 4)[1].celulas
        == 0
    )
    pior = executar_contando("knapsack", instancia_do_cenario(chunks, "knapsack", "pior", 1024, "x"), 1024, 4)[1]
    assert pior.celulas > 0.98 * 200 * 1024
    assert instancia_do_cenario(chunks, "greedy", "tipico", 1024, "x") == chunks


def test_instancias_sao_aninhadas_e_deterministicas(exemplos_sinteticos) -> None:
    extras = extras_do_caso(exemplos_sinteticos, 0, 2026)
    assert extras == extras_do_caso(exemplos_sinteticos, 0, 2026)
    pequena = montar_instancia(exemplos_sinteticos[0], extras, 20)
    grande = montar_instancia(exemplos_sinteticos[0], extras, 50)
    assert len(pequena) == 20 and len(grande) == 50
    assert [c.id for c in grande[:20]] == [c.id for c in pequena]
    assert len({c.id for c in grande}) == 50
    assert montar_instancia(exemplos_sinteticos[0], extras, 5) == list(exemplos_sinteticos[0].chunks[:5])
    with pytest.raises(ValueError):
        montar_instancia(exemplos_sinteticos[0], extras, 10_000)


def test_ajustes_recuperam_expoentes_conhecidos() -> None:
    xs = [10, 20, 50, 100, 200]
    assert ajuste_potencia(xs, [3 * x**2 for x in xs])["expoente"] == pytest.approx(2.0)
    n = [10, 20, 40, 10, 20, 40]
    t = [100, 100, 100, 800, 800, 800]
    ajuste = ajuste_n_e_t(n, t, [5 * a * b for a, b in zip(n, t, strict=True)])
    assert ajuste["expoente_n"] == pytest.approx(1.0) and ajuste["expoente_T"] == pytest.approx(1.0)
    assert ajuste["r2"] == pytest.approx(1.0)
    assert ajuste_exponencial([4, 6, 8, 10], [7 * 2**k for k in (4, 6, 8, 10)])["base"] == pytest.approx(2.0)
    assert 1.05 < expoente_efetivo_nlogn([100, 200, 400, 800]) < 1.2


def test_cronometrar_mede_tempo_positivo_e_religa_o_coletor() -> None:
    assert gc.isenabled()
    tempo, repeticoes = cronometrar(lambda: sum(range(1000)))
    assert tempo > 0 and repeticoes >= 1
    assert gc.isenabled()


def test_varreduras_pequenas(exemplos_sinteticos, tmp_path: Path) -> None:
    medidas, memoria = executar_varreduras(exemplos_sinteticos[:2], exemplos_sinteticos, GRADE_TESTE, 4)
    por_varredura = {
        v: [m for m in medidas if m["varredura"] == v] for v in ("principal", "forca_bruta", "cenarios", "escala")
    }
    assert len(por_varredura["principal"]) == 2 * 2 * 2 * len(ALGORITMOS)
    assert {m["algoritmo"] for m in por_varredura["forca_bruta"]} == {"forca_bruta", "knapsack"}
    assert len(por_varredura["cenarios"]) == 2 * len(ALGORITMOS) * 3
    assert len(por_varredura["escala"]) == 2 * len(ALGORITMOS)
    assert all(m["tempo_ns"] > 0 and m["operacoes"] > 0 for m in medidas)
    assert all(
        m["razao_otimo"] == pytest.approx(1.0) for m in por_varredura["principal"] if m["algoritmo"] == "knapsack"
    )
    assert len(memoria) == 2 * 2 * len(ALGORITMOS) + 2 * 2
    gravar_analise(tmp_path / "assintotica-teste", medidas, memoria, {"casos": 2})
    assert (tmp_path / "assintotica-teste" / "medidas.jsonl").stat().st_size > 0


def test_resumo_dos_ajustes_ignora_eixos_com_um_so_valor(exemplos_sinteticos) -> None:
    grade = Grade(
        n=(5, 20),
        t=(512,),
        casos_memoria=0,
        n_forca_bruta=(4, 6, 8),
        casos_forca_bruta=1,
        casos_cenarios=0,
        casos_escala=0,
    )
    medidas, _ = executar_varreduras(exemplos_sinteticos[:1], exemplos_sinteticos, grade, 4)
    ajustes = resumo_dos_ajustes(medidas)
    assert not [a for a in ajustes if "expoente_n" in a]  # um só T: expoentes de n·T não identificáveis
    assert [a for a in ajustes if a["algoritmo"] == "forca_bruta"]
