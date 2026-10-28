"""O painel Streamlit roda sem erro em todas as abas (AppTest, sem navegador)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAINEL = str(Path(__file__).resolve().parents[1] / "src" / "rag_knapsack" / "painel.py")


def abrir(monkeypatch: pytest.MonkeyPatch, pasta: Path) -> AppTest:
    monkeypatch.setenv("PAINEL_RESULTADOS", str(pasta))
    return AppTest.from_file(PAINEL, default_timeout=120).run()


def test_painel_com_resultados(monkeypatch: pytest.MonkeyPatch, pasta_experimento: Path) -> None:
    app = abrir(monkeypatch, pasta_experimento.parent)
    assert not app.exception
    assert [aba.label for aba in app.tabs] == [
        "Visão geral",
        "Orçamento × qualidade",
        "Otimalidade e latência",
        "Perguntas",
        "Exemplo de Brasília",
    ]
    # Aba de perguntas: resposta ideal e a resposta de cada algoritmo.
    textos = " ".join(elemento.value for elemento in app.markdown)
    assert "Resposta ideal" in textos and "Knapsack DP" in textos

    app.sidebar.select_slider[0].set_value(256).run()
    app.toggle(key="so_divergencias").set_value(True).run()
    app.sidebar.multiselect[0].set_value(["greedy", "knapsack"]).run()
    for metrica in ("f1", "tokens_usados", "custo_medio_usd"):
        app.segmented_control(key="metrica_orcamento").set_value(metrica).run()
    app.slider(key="t_brasilia").set_value(150).run()
    assert not app.exception


def test_painel_sem_resultados_mostra_como_gerar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = abrir(monkeypatch, tmp_path)
    assert not app.exception
    assert "uv run task avaliar" in app.warning[0].value
    assert len(app.metric) == 4  # o exemplo de Brasília funciona sem dados
