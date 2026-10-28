"""Experimento comparativo: Top-K, Greedy e Knapsack DP nas mesmas perguntas e orçamentos.

Exemplos:
    uv run task avaliar                                  # 30 perguntas, T = 256, 512 e 1024
    uv run task avaliar --n 100 --orcamentos 128 256 512 1024
    uv run task avaliar --sem-llm                        # só seleção, otimalidade e desempenho

Cada execução grava uma pasta em resultados/ com:
    linhas.jsonl       uma linha por (pergunta, algoritmo, orçamento), com todas as métricas
    perguntas.jsonl    texto das perguntas e dos parágrafos candidatos (para o painel)
    desempenho.json    tempo e memória da seleção (PerformanceEval do agno)
    execucao.json      configuração, gasto real com a API e resumo por algoritmo
O painel (`uv run task painel`) lê essas pastas.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from rag_knapsack.config import Config, inteiro_positivo, tem_chave_openai
from rag_knapsack.dados import CAMINHO_AMOSTRA, Exemplo, caminho_dev, carregar_exemplos
from rag_knapsack.evals import eval_desempenho, resumir_linhas
from rag_knapsack.gerador import Gerador
from rag_knapsack.juiz import Juiz
from rag_knapsack.pipeline import (
    Resultado,
    criar_gerador,
    criar_juiz,
    gerar_respostas,
    gravar_jsonl,
    julgar,
    selecionar,
)
from rag_knapsack.recuperador import criar_recuperador
from rag_knapsack.seletores import ALGORITMOS, Seletor, criar_seletor

ORCAMENTOS_PADRAO = (256, 512, 1024)


def executar(
    exemplos: Sequence[Exemplo],
    seletores: Sequence[Seletor],
    orcamentos: Sequence[int],
    gerador: Gerador | None = None,
    juiz: Juiz | None = None,
    workers: int = 8,
) -> list[Resultado]:
    """Roda cada seletor em cada orçamento sobre as mesmas perguntas (já pontuadas)."""
    resultados = [
        resultado
        for orcamento in orcamentos
        for seletor in seletores
        for resultado in selecionar(exemplos, seletor, orcamento)
    ]
    if gerador is not None:
        gerar_respostas(resultados, gerador, workers)
        if juiz is not None:
            julgar(resultados, juiz, workers)
    return resultados


def resumo_por_grupo(linhas: Sequence[dict]) -> list[dict]:
    """Uma linha de resumo por (orçamento, algoritmo)."""
    grupos: dict[tuple[int, str], list[dict]] = {}
    for linha in linhas:
        grupos.setdefault((linha["orcamento"], linha["algoritmo"]), []).append(linha)
    ordem = {nome: i for i, nome in enumerate(ALGORITMOS)}
    return [
        {"orcamento": orcamento, "algoritmo": algoritmo, **resumir_linhas(grupo)}
        for (orcamento, algoritmo), grupo in sorted(grupos.items(), key=lambda g: (g[0][0], ordem.get(g[0][1], 99)))
    ]


def formatar_tabela(resumos: Sequence[dict]) -> str:
    colunas = [
        ("orcamento", "T", "{}"),
        ("algoritmo", "algoritmo", "{}"),
        ("acuracia_juiz", "juiz", "{:.2f}"),
        ("em", "EM", "{:.2f}"),
        ("f1", "F1", "{:.2f}"),
        ("recall_suporte", "recall", "{:.2f}"),
        ("suporte_completo", "2 sup.", "{:.2f}"),
        ("taxa_estouro", "estouro", "{:.2f}"),
        ("tokens_usados", "tokens", "{:.0f}"),
        ("razao_otimo", "v/ótimo", "{:.3f}"),
    ]
    presentes = [coluna for coluna in colunas if any(coluna[0] in resumo for resumo in resumos)]
    tabela = [[titulo for _, titulo, _ in presentes]]
    for resumo in resumos:
        tabela.append([fmt.format(resumo[campo]) if campo in resumo else "-" for campo, _, fmt in presentes])
    larguras = [max(len(linha[i]) for linha in tabela) for i in range(len(presentes))]
    return "\n".join(
        "  ".join(celula.rjust(largura) for celula, largura in zip(linha, larguras, strict=True)) for linha in tabela
    )


def _registro_pergunta(exemplo: Exemplo) -> dict:
    return {
        "id": exemplo.id,
        "pergunta": exemplo.pergunta,
        "resposta": exemplo.resposta,
        "tipo": exemplo.tipo,
        "chunks": [
            {
                "id": c.id,
                "titulo": c.titulo,
                "texto": c.texto,
                "tokens": c.tokens,
                "score": c.score,
                "de_suporte": c.de_suporte,
            }
            for c in exemplo.chunks
        ],
    }


def gravar_experimento(
    pasta: Path, linhas: Sequence[dict], exemplos: Sequence[Exemplo], desempenho: Sequence[dict], execucao: dict
) -> None:
    """Grava a pasta que o painel lê (linhas, perguntas, desempenho e execução)."""
    gravar_jsonl(linhas, pasta / "linhas.jsonl")
    gravar_jsonl([_registro_pergunta(exemplo) for exemplo in exemplos], pasta / "perguntas.jsonl")
    (pasta / "desempenho.json").write_text(json.dumps(list(desempenho), indent=2), encoding="utf-8")
    execucao = {**execucao, "resumo": resumo_por_grupo(linhas)}
    (pasta / "execucao.json").write_text(json.dumps(execucao, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=inteiro_positivo, default=30, help="quantidade de perguntas (padrão: 30)")
    parser.add_argument(
        "--orcamentos", type=inteiro_positivo, nargs="+", default=list(ORCAMENTOS_PADRAO), help="valores de T"
    )
    parser.add_argument("--algoritmos", nargs="+", choices=ALGORITMOS, default=list(ALGORITMOS))
    parser.add_argument("--fonte", choices=("auto", "dev", "amostra"), default="auto")
    parser.add_argument("--k", type=inteiro_positivo, default=None, help="k do Top-K (padrão: TOP_K do .env)")
    parser.add_argument("--sem-llm", action="store_true", help="não gera respostas (só seleção e desempenho)")
    parser.add_argument("--sem-juiz", action="store_true", help="gera as respostas, mas não chama o juiz")
    parser.add_argument("--sem-cache", action="store_true", help="ignora as respostas do LLM já guardadas")
    parser.add_argument("--workers", type=inteiro_positivo, default=8, help="chamadas simultâneas à API")
    parser.add_argument("--iteracoes", type=inteiro_positivo, default=20, help="repetições do PerformanceEval")
    args = parser.parse_args(argv)

    cfg = Config.do_ambiente()
    usa_llm = not args.sem_llm
    if (usa_llm or cfg.precisa_chave_openai) and not tem_chave_openai():
        print("Defina OPENAI_API_KEY no .env, ou rode com EMBEDDER=fastembed e --sem-llm.", file=sys.stderr)
        return 1

    dev = caminho_dev(cfg)
    if args.fonte == "dev" and not dev.exists():
        sys.exit("Dataset não encontrado. Baixe com: uv run task dados")
    fonte = CAMINHO_AMOSTRA if args.fonte == "amostra" or (args.fonte == "auto" and not dev.exists()) else dev
    k = args.k or cfg.top_k
    seletores = [criar_seletor(nome, k=k) for nome in args.algoritmos]
    orcamentos = sorted(set(args.orcamentos))

    exemplos = carregar_exemplos(fonte, args.n)
    print(f"{len(exemplos)} perguntas de {fonte.name} | {', '.join(args.algoritmos)} | T = {orcamentos} | k = {k}")
    recuperador = criar_recuperador(cfg)
    exemplos = [recuperador.preparar(exemplo) for exemplo in exemplos]
    recuperador.salvar_cache()

    gerador = juiz = None
    if usa_llm:
        gerador = criar_gerador(cfg, usar_cache=not args.sem_cache)
        juiz = None if args.sem_juiz else criar_juiz(cfg, usar_cache=not args.sem_cache)
        print(f"Gerando com {cfg.modelo_gerador} e julgando com {cfg.modelo_juiz if juiz else '(sem juiz)'}...")

    try:
        resultados = executar(exemplos, seletores, orcamentos, gerador, juiz, args.workers)
    finally:
        for componente in (gerador, juiz):
            if componente is not None:
                componente.cache.salvar()

    print("Medindo o desempenho da seleção (PerformanceEval)...")
    desempenho = eval_desempenho(seletores, exemplos, orcamentos, args.iteracoes)

    linhas = [resultado.como_dict(cfg) for resultado in resultados]
    agora = datetime.now()
    gasto = {}
    precos = {
        "gerador": (cfg.preco_entrada_mtok, cfg.preco_saida_mtok),
        "juiz": (cfg.preco_entrada_juiz_mtok, cfg.preco_saida_juiz_mtok),
    }
    for nome, componente in (("gerador", gerador), ("juiz", juiz)):
        if componente is not None:
            gasto[nome] = {
                "chamadas_api": componente.gasto.chamadas,
                "tokens_entrada": componente.gasto.tokens_entrada,
                "tokens_saida": componente.gasto.tokens_saida,
                "custo_estimado_usd": componente.gasto.custo_usd(*precos[nome]),
            }
    pasta = cfg.dir_resultados / f"avaliacao-{agora:%Y%m%d-%H%M%S}"
    gravar_experimento(
        pasta,
        linhas,
        exemplos,
        desempenho,
        {
            "data": agora.isoformat(timespec="seconds"),
            "fonte": fonte.name,
            "perguntas": len(exemplos),
            "algoritmos": args.algoritmos,
            "orcamentos": orcamentos,
            "k": k,
            "modelo_gerador": cfg.modelo_gerador if usa_llm else None,
            "esforco_gerador": cfg.esforco_gerador if usa_llm else None,
            "modelo_juiz": cfg.modelo_juiz if juiz else None,
            "esforco_juiz": cfg.esforco_juiz if juiz else None,
            "embedder": f"{cfg.embedder}/{cfg.modelo_embedding}",
            "tokenizador": recuperador.contador.nome,
            "precos_usd_por_mtok": {"entrada": cfg.preco_entrada_mtok, "saida": cfg.preco_saida_mtok},
            "gasto_api": gasto,
        },
    )
    resumos = resumo_por_grupo(linhas)

    print(formatar_tabela(resumos))
    for nome, dados in gasto.items():
        print(f"{nome}: {dados['chamadas_api']} chamadas à API, ~US$ {dados['custo_estimado_usd']:.4f}")
    erros = [linha for linha in linhas if linha["erro"]]
    if erros:
        print(f"{len(erros)} linhas com erro (detalhes em linhas.jsonl)", file=sys.stderr)
    print(f"Resultados em {pasta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
