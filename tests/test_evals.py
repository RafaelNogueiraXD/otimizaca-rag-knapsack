"""Testes das funções de avaliação, do cache e do experimento comparativo (sem rede)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import GeradorFalso, JuizFalso

from rag_knapsack.cache import CacheJson
from rag_knapsack.config import Config
from rag_knapsack.evals import eval_desempenho, eval_otimalidade, eval_resposta, resumir_linhas
from rag_knapsack.experimento import executar, resumo_por_grupo
from rag_knapsack.metricas import exact_match, f1
from rag_knapsack.pipeline import gerar_respostas, julgar, selecionar
from rag_knapsack.seletores import ALGORITMOS, Knapsack, TopK, criar_seletor


@pytest.mark.parametrize(
    ("predita", "referencia", "em", "f1_esperado"),
    [
        ("The Eiffel Tower.", "Eiffel Tower", 1.0, 1.0),  # artigos e pontuação não contam
        ("yes", "no", 0.0, 0.0),
        ("Yes", "yes", 1.0, 1.0),
        ("Kevin Spacey", "Spacey", 0.0, 2 / 3),
        ("no", "no idea at all", 0.0, 0.0),  # sim/não só vale se for igual
    ],
)
def test_em_e_f1_seguem_o_script_oficial(predita: str, referencia: str, em: float, f1_esperado: float) -> None:
    assert exact_match(predita, referencia) == em
    assert f1(predita, referencia) == pytest.approx(f1_esperado)


def test_eval_resposta_marca_unknown(exemplos_sinteticos) -> None:
    exemplo = exemplos_sinteticos[0]
    assert eval_resposta("Unknown.", exemplo)["sem_resposta"] is True
    assert eval_resposta(exemplo.resposta, exemplo)["em"] == 1.0


def test_eval_otimalidade(exemplos_sinteticos) -> None:
    exemplo = exemplos_sinteticos[0]
    otima = eval_otimalidade(Knapsack().selecionar(exemplo.chunks, 300), exemplo)
    assert otima["razao_otimo"] == pytest.approx(1.0) and otima["atingiu_otimo"]

    estouro = TopK(k=10).selecionar(exemplo.chunks, 50)
    resultado = eval_otimalidade(estouro, exemplo)
    assert estouro.estourou_orcamento and not resultado["atingiu_otimo"]


def test_resumo_ignora_selecoes_invalidas_na_razao_do_otimo() -> None:
    linhas = [
        {
            "tokens_usados": 100,
            "ocupacao": 0.5,
            "valor_total": 1.0,
            "recall_suporte": 1.0,
            "precisao_suporte": 0.5,
            "suporte_completo": True,
            "estourou_orcamento": False,
            "razao_otimo": 0.8,
            "atingiu_otimo": False,
        },
        {
            "tokens_usados": 300,
            "ocupacao": 1.5,
            "valor_total": 2.0,
            "recall_suporte": 1.0,
            "precisao_suporte": 0.5,
            "suporte_completo": True,
            "estourou_orcamento": True,
            "razao_otimo": 1.4,
            "atingiu_otimo": False,
        },
    ]
    resumo = resumir_linhas(linhas)
    assert resumo["razao_otimo"] == pytest.approx(0.8)
    assert resumo["taxa_estouro"] == pytest.approx(0.5)
    assert "acuracia_juiz" not in resumo and "em" not in resumo


def test_falha_da_api_conta_como_resposta_errada() -> None:
    base = {
        "tokens_usados": 100,
        "ocupacao": 0.5,
        "valor_total": 1.0,
        "recall_suporte": 1.0,
        "precisao_suporte": 0.5,
        "suporte_completo": True,
        "estourou_orcamento": False,
    }
    linhas = [
        {
            **base,
            "resposta_gerada": "1755",
            "em": 1.0,
            "f1": 1.0,
            "sem_resposta": False,
            "tokens_entrada": 600,
            "tokens_saida": 10,
            "custo_usd": 0.0001,
            "juiz_correto": True,
            "erro": None,
        },
        {**base, "resposta_gerada": None, "erro": "gpt-6-luna não respondeu: 'Connection error.'"},
    ]
    resumo = resumir_linhas(linhas)
    assert resumo["em"] == pytest.approx(0.5) and resumo["f1"] == pytest.approx(0.5)
    assert resumo["acuracia_juiz"] == pytest.approx(0.5)
    assert resumo["erros"] == 1 and resumo["respondidas"] == 1


def test_contextos_iguais_geram_uma_chamada_so(exemplos_sinteticos) -> None:
    exemplos = exemplos_sinteticos[:3]
    gerador, juiz = GeradorFalso(exemplos), JuizFalso()
    # O mesmo seletor em dois orçamentos grandes escolhe o mesmo contexto: metade das chamadas.
    resultados = selecionar(exemplos, Knapsack(), 5000) + selecionar(exemplos, Knapsack(), 6000)
    gerar_respostas(resultados, gerador)
    julgar(resultados, juiz)
    assert gerador.chamadas == 3 and juiz.chamadas == 3
    assert all(resultado.resposta is not None and resultado.veredito is not None for resultado in resultados)


def test_experimento_produz_uma_linha_por_pergunta_algoritmo_e_orcamento(cfg: Config, exemplos_sinteticos) -> None:
    seletores = [criar_seletor(nome, k=4) for nome in ALGORITMOS]
    resultados = executar(exemplos_sinteticos, seletores, [256, 512], GeradorFalso(exemplos_sinteticos), JuizFalso())
    linhas = [resultado.como_dict(cfg) for resultado in resultados]
    assert len(linhas) == len(exemplos_sinteticos) * len(ALGORITMOS) * 2
    resumos = resumo_por_grupo(linhas)
    assert [(r["orcamento"], r["algoritmo"]) for r in resumos[:4]] == [(256, a) for a in ALGORITMOS]
    knapsack = next(r for r in resumos if r["algoritmo"] == "knapsack" and r["orcamento"] == 512)
    assert knapsack["razao_otimo"] == pytest.approx(1.0) and knapsack["taxa_estouro"] == 0
    # O dublê só acerta com todo o suporte no contexto: acurácia = fração com suporte completo.
    for resumo in resumos:
        assert resumo["acuracia_juiz"] == pytest.approx(resumo["suporte_completo"])


def test_eval_desempenho_usa_o_performance_eval(exemplos_sinteticos) -> None:
    linhas = eval_desempenho([Knapsack(), TopK(k=4)], exemplos_sinteticos[:2], [256], iteracoes=2)
    assert {linha["algoritmo"] for linha in linhas} == {"knapsack", "topk"}
    assert all(linha["ms_por_selecao"] > 0 and linha["memoria_pico_mib"] >= 0 for linha in linhas)


def test_pasta_do_experimento_tem_o_formato_do_painel(pasta_experimento: Path) -> None:
    for arquivo in ("linhas.jsonl", "perguntas.jsonl", "desempenho.json", "execucao.json"):
        assert (pasta_experimento / arquivo).exists()
    execucao = json.loads((pasta_experimento / "execucao.json").read_text(encoding="utf-8"))
    assert execucao["resumo"] and execucao["algoritmos"] == list(ALGORITMOS)


def test_cache_json_persiste(tmp_path: Path) -> None:
    caminho = tmp_path / "cache.json"
    cache = CacheJson(caminho)
    cache.set("a", {"texto": "1755"})
    cache.salvar()
    assert CacheJson(caminho).get("a") == {"texto": "1755"}
    assert "b" not in CacheJson(caminho)
