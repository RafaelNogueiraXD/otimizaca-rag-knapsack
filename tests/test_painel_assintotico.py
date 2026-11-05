"""O painel da análise assintótica roda sem erro em todas as abas (AppTest, sem navegador)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_knapsack.assintotica import Grade, executar_varreduras, gravar_analise, maquina, resumo_dos_ajustes

PAINEL = str(Path(__file__).resolve().parents[1] / "src" / "rag_knapsack" / "painel_assintotico.py")
GRADE = Grade(
    n=(5, 10, 20, 40),
    t=(128, 256, 512, 1024),
    casos_memoria=2,
    n_forca_bruta=(4, 6, 8, 10),
    casos_forca_bruta=2,
    n_cenarios=(10, 20, 40),
    casos_cenarios=2,
    fatores_escala=(1, 2, 4),
    n_escala=20,
    casos_escala=2,
)
ABAS = [
    "Resumo",
    "Teoria",
    "Tempo × n",
    "Tempo × T",
    "Operações",
    "Mapa n × T",
    "Força bruta × DP",
    "Melhor × pior caso",
    "Pseudo-polinomial",
    "Memória",
    "Qualidade × tamanho",
    "Metodologia e dados",
]


@pytest.fixture(scope="module")
def pasta_analise(tmp_path_factory: pytest.TempPathFactory, exemplos_sinteticos) -> Path:
    medidas, memoria = executar_varreduras(exemplos_sinteticos[:3], exemplos_sinteticos, GRADE, 4)
    raiz = tmp_path_factory.mktemp("resultados")
    gravar_analise(
        raiz / "assintotica-20260101-000000",
        medidas,
        memoria,
        {
            "data": "2026-01-01T00:00:00",
            "fonte": "amostra",
            "casos": 3,
            "pool": len(exemplos_sinteticos),
            "n": list(GRADE.n),
            "T": list(GRADE.t),
            "k": 4,
            "grade": GRADE.como_dict(),
            "regime": {"n_minimo": 10, "t_minimo": 256, "n_minimo_forca_bruta": 6},
            "semente": 2026,
            "embedder": "sintetico",
            "tokenizador": "len/4",
            "maquina": maquina(),
            "duracao_s": 1.0,
            "ajustes": resumo_dos_ajustes(medidas, 10, 256, 6),
        },
    )
    return raiz


def abrir(monkeypatch: pytest.MonkeyPatch, pasta: Path) -> AppTest:
    monkeypatch.setenv("PAINEL_RESULTADOS", str(pasta))
    return AppTest.from_file(PAINEL, default_timeout=180).run()


def test_painel_assintotico_renderiza_todas_as_abas(monkeypatch: pytest.MonkeyPatch, pasta_analise: Path) -> None:
    app = abrir(monkeypatch, pasta_analise)
    assert not app.exception, [e.value for e in app.exception]
    assert [aba.label for aba in app.tabs] == ABAS
    app.segmented_control(key="medida_n").set_value("operacoes").run()
    app.select_slider(key="fixo_n").set_value(1024).run()
    app.select_slider(key="fixo_T").set_value(40).run()
    app.segmented_control(key="algoritmo_mapa").set_value("greedy").run()
    app.segmented_control(key="medida_cenarios").set_value("tempo_us").run()
    app.segmented_control(key="medida_escala").set_value("tempo_us").run()
    app.select_slider(key="n_minimo_forca").set_value(4).run()
    app.sidebar.multiselect[0].set_value(["knapsack", "greedy"]).run()
    assert not app.exception, [e.value for e in app.exception]
    app.sidebar.multiselect[0].set_value(["knapsack"]).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("heurística" in info.value for info in app.info)


def test_painel_assintotico_sem_dados(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = abrir(monkeypatch, tmp_path)
    assert not app.exception
    assert "uv run task assintotica" in app.warning[0].value
