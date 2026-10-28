"""Funções de avaliação (evals).

Cada `eval_*` olha uma dimensão de uma pergunta respondida e devolve um dicionário de
métricas; `resumir_linhas` agrega essas linhas. Ligação com as hipóteses da proposta:

- eval_selecao       contexto escolhido: tokens, estouro de T, parágrafos de suporte
- eval_otimalidade   valor da seleção / valor ótimo da mochila no mesmo T       (H2)
- eval_resposta      EM e F1 oficiais do HotpotQA                               (H1)
- eval_juiz          veredito do Agent-as-a-judge                               (H1)
- eval_custo         tokens cobrados pela API, custo e tempo de geração
- eval_desempenho    tempo e memória da seleção, com o PerformanceEval do agno  (H3)
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import cycle
from statistics import fmean

from agno.eval.performance import PerformanceEval

from rag_knapsack.config import Config
from rag_knapsack.dados import Exemplo
from rag_knapsack.gerador import RespostaGerada
from rag_knapsack.juiz import Veredito
from rag_knapsack.metricas import exact_match, f1, precisao_suporte, recall_suporte, sem_resposta, suporte_completo
from rag_knapsack.seletores import Knapsack, Selecao, Seletor

TOLERANCIA = 1e-9


def eval_selecao(selecao: Selecao, exemplo: Exemplo) -> dict:
    return {
        "tokens_usados": selecao.tokens_usados,
        "ocupacao": selecao.ocupacao,
        "estourou_orcamento": selecao.estourou_orcamento,
        "valor_total": selecao.valor_total,
        "recall_suporte": recall_suporte(selecao, exemplo),
        "precisao_suporte": precisao_suporte(selecao, exemplo),
        "suporte_completo": suporte_completo(selecao, exemplo),
    }


def eval_otimalidade(selecao: Selecao, exemplo: Exemplo) -> dict:
    """Compara o objetivo da seleção (soma dos scores) com o ótimo exato no mesmo orçamento.

    Uma seleção que estoura T (Top-K padrão) pode passar de 1, mas aí não é uma solução
    válida da mochila; por isso `atingiu_otimo` exige respeitar o orçamento.
    """
    otimo = Knapsack().selecionar(exemplo.chunks, selecao.orcamento).valor_total
    razao = selecao.valor_total / otimo if otimo > 0 else 1.0
    return {
        "valor_otimo": otimo,
        "razao_otimo": razao,
        "atingiu_otimo": not selecao.estourou_orcamento and selecao.valor_total >= otimo - TOLERANCIA,
    }


def eval_resposta(resposta: str, exemplo: Exemplo) -> dict:
    return {
        "em": exact_match(resposta, exemplo.resposta),
        "f1": f1(resposta, exemplo.resposta),
        "sem_resposta": sem_resposta(resposta),
    }


def eval_juiz(veredito: Veredito) -> dict:
    return {"juiz_correto": veredito.correto, "juiz_justificativa": veredito.justificativa}


def eval_custo(resposta: RespostaGerada, cfg: Config) -> dict:
    custo = (resposta.tokens_entrada * cfg.preco_entrada_mtok + resposta.tokens_saida * cfg.preco_saida_mtok) / 1e6
    return {
        "tokens_entrada": resposta.tokens_entrada,
        "tokens_saida": resposta.tokens_saida,
        "tokens_raciocinio": resposta.tokens_raciocinio,
        "custo_usd": custo,
        "duracao_geracao_s": resposta.duracao_s,
    }


def eval_desempenho(
    seletores: Sequence[Seletor],
    exemplos: Sequence[Exemplo],
    orcamentos: Sequence[int],
    iteracoes: int = 20,
) -> list[dict]:
    """Mede o tempo e o pico de memória de uma seleção com o `PerformanceEval` do agno.

    Cada chamada medida seleciona o contexto de uma pergunta, percorrendo as perguntas em
    ciclo, então tempo e memória são de uma seleção (e não crescem com o número de perguntas).
    """
    linhas = []
    for seletor in seletores:
        for orcamento in orcamentos:
            ciclo = cycle(exemplos)
            resultado = PerformanceEval(
                func=lambda s=seletor, t=orcamento, c=ciclo: s.selecionar(next(c).chunks, t),
                name=f"selecao-{seletor.nome}-T{orcamento}",
                num_iterations=max(iteracoes, len(exemplos)),
                warmup_runs=3,
                show_spinner=False,
                telemetry=False,
            ).run()
            linhas.append(
                {
                    "algoritmo": seletor.nome,
                    "orcamento": orcamento,
                    "ms_por_selecao": resultado.avg_run_time * 1000,
                    "ms_por_selecao_mediana": resultado.median_run_time * 1000,
                    "memoria_pico_mib": resultado.avg_memory_usage,
                }
            )
    return linhas


def _media(linhas: Sequence[dict], campo: str) -> float:
    return fmean(linha[campo] for linha in linhas)


def resumir_linhas(linhas: Sequence[dict]) -> dict[str, float]:
    """Agrega as linhas produzidas pelas funções `eval_*` (uma por pergunta)."""
    resumo: dict[str, float] = {"perguntas": len(linhas)}
    for campo in ("tokens_usados", "ocupacao", "valor_total", "recall_suporte", "precisao_suporte", "suporte_completo"):
        resumo[campo] = _media(linhas, campo)
    resumo["taxa_estouro"] = _media(linhas, "estourou_orcamento")
    if all("razao_otimo" in linha for linha in linhas):
        validas = [linha for linha in linhas if not linha["estourou_orcamento"]]
        if validas:
            resumo["razao_otimo"] = _media(validas, "razao_otimo")
        resumo["atingiu_otimo"] = _media(linhas, "atingiu_otimo")
    if all("latencia_selecao_ms" in linha for linha in linhas):
        resumo["latencia_selecao_ms"] = _media(linhas, "latencia_selecao_ms")

    # Uma falha da API conta como resposta errada: assim todos os algoritmos têm o mesmo n e
    # quem falhou mais não fica com a média inflada. O número de erros aparece no resumo.
    tentadas = [linha for linha in linhas if linha.get("resposta_gerada") is not None or linha.get("erro")]
    respondidas = [linha for linha in tentadas if linha.get("resposta_gerada") is not None]
    if tentadas:
        resumo["respondidas"] = len(respondidas)
        resumo["em"] = sum(linha.get("em") or 0.0 for linha in tentadas) / len(tentadas)
        resumo["f1"] = sum(linha.get("f1") or 0.0 for linha in tentadas) / len(tentadas)
        resumo["taxa_sem_resposta"] = sum(bool(linha.get("sem_resposta")) for linha in tentadas) / len(tentadas)
    if respondidas:
        resumo["tokens_entrada_api"] = _media(respondidas, "tokens_entrada")
        resumo["tokens_saida_api"] = _media(respondidas, "tokens_saida")
        resumo["custo_total_usd"] = sum(linha["custo_usd"] for linha in respondidas)
    # Com o juiz ligado, pergunta sem veredito (geração ou juiz falharam) conta como incorreta.
    if any(linha.get("juiz_correto") is not None for linha in tentadas):
        resumo["acuracia_juiz"] = sum(linha.get("juiz_correto") is True for linha in tentadas) / len(tentadas)
    resumo["erros"] = sum(linha.get("erro") is not None for linha in linhas)
    return resumo


def formatar_resumo(resumo: dict[str, float]) -> str:
    largura = max(len(nome) for nome in resumo)
    linhas = []
    for nome, valor in resumo.items():
        if isinstance(valor, int):
            texto = str(valor)
        elif nome == "custo_total_usd":
            texto = f"US$ {valor:.6f}"
        else:
            texto = f"{valor:.3f}"
        linhas.append(f"  {nome:<{largura}}  {texto}")
    return "\n".join(linhas)
