"""Roda o experimento de seleção de contexto no HotpotQA para um algoritmo.

Exemplos:
    uv run task knapsack                  # 20 perguntas, T=512
    uv run task greedy --n 100 --orcamento 256
    uv run task topk --sem-llm            # só a seleção (sem gerar respostas)

Para comparar os três algoritmos lado a lado, use `uv run task avaliar` (experimento.py).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rag_knapsack.config import Config, inteiro_positivo, tem_chave_openai
from rag_knapsack.dados import CAMINHO_AMOSTRA, caminho_dev, carregar_exemplos
from rag_knapsack.evals import formatar_resumo, resumir_linhas
from rag_knapsack.pipeline import criar_gerador, criar_juiz, gerar_respostas, gravar_jsonl, julgar, selecionar
from rag_knapsack.recuperador import criar_recuperador
from rag_knapsack.seletores import ALGORITMOS, criar_seletor


def _fonte_dos_dados(fonte: str, cfg: Config) -> Path:
    dev = caminho_dev(cfg)
    if fonte == "amostra" or (fonte == "auto" and not dev.exists()):
        return CAMINHO_AMOSTRA
    if not dev.exists():
        sys.exit("Dataset não encontrado. Baixe com: uv run task dados")
    return dev


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algoritmo", choices=ALGORITMOS, required=True)
    parser.add_argument("--n", type=inteiro_positivo, default=20, help="quantidade de perguntas (padrão: 20)")
    parser.add_argument(
        "--fonte",
        choices=("auto", "dev", "amostra"),
        default="auto",
        help="dev = HotpotQA baixado; amostra = as 10 perguntas dos testes; auto = dev se existir",
    )
    parser.add_argument("--k", type=inteiro_positivo, default=None, help="k do Top-K (padrão: TOP_K do .env)")
    parser.add_argument(
        "--orcamento", type=inteiro_positivo, default=None, help="orçamento T em tokens (padrão: ORCAMENTO_TOKENS)"
    )
    parser.add_argument("--sem-llm", action="store_true", help="só seleção e métricas de suporte, sem gerar respostas")
    parser.add_argument("--sem-juiz", action="store_true", help="gera as respostas, mas não chama o juiz")
    parser.add_argument("--workers", type=inteiro_positivo, default=4, help="chamadas simultâneas à API")
    parser.add_argument("--saida", type=Path, default=None, help="arquivo JSONL de resultados")
    args = parser.parse_args(argv)

    cfg = Config.do_ambiente()
    orcamento = args.orcamento or cfg.orcamento_tokens
    seletor = criar_seletor(args.algoritmo, k=args.k or cfg.top_k)
    usa_llm = not args.sem_llm
    if (usa_llm or cfg.precisa_chave_openai) and not tem_chave_openai():
        print(
            "Defina OPENAI_API_KEY no arquivo .env (veja .env.example). "
            "Sem chave, dá para rodar só a seleção com EMBEDDER=fastembed e --sem-llm.",
            file=sys.stderr,
        )
        return 1

    fonte = _fonte_dos_dados(args.fonte, cfg)
    exemplos = carregar_exemplos(fonte, args.n)
    print(
        f"{len(exemplos)} perguntas de {fonte.name} | {seletor.nome} | T={orcamento} | embedder={cfg.modelo_embedding}"
    )

    recuperador = criar_recuperador(cfg)
    exemplos = [recuperador.preparar(exemplo) for exemplo in exemplos]
    recuperador.salvar_cache()
    resultados = selecionar(exemplos, seletor, orcamento)

    if usa_llm:
        print(f"Gerando respostas com {cfg.modelo_gerador} (reasoning_effort={cfg.esforco_gerador})...")
        gerador = criar_gerador(cfg)
        gerar_respostas(resultados, gerador, args.workers)
        gerador.cache.salvar()
        if not args.sem_juiz:
            print(f"Julgando com {cfg.modelo_juiz} (Agent-as-a-judge)...")
            juiz = criar_juiz(cfg)
            julgar(resultados, juiz, args.workers)
            juiz.cache.salvar()

    saida = args.saida or cfg.dir_resultados / f"{seletor.nome}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    linhas = [resultado.como_dict(cfg) for resultado in resultados]
    gravar_jsonl(linhas, saida)
    print(formatar_resumo(resumir_linhas(linhas)))
    print(f"Resultados por pergunta: {saida}")
    for resultado in resultados:
        if resultado.erro:
            print(f"  erro em {resultado.exemplo.id}: {resultado.erro}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
