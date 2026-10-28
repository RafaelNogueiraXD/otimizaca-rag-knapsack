"""Pipeline de uma pergunta: seleção -> geração -> julgamento, e o registro das métricas."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rag_knapsack.cache import CacheJson
from rag_knapsack.config import Config
from rag_knapsack.dados import Exemplo
from rag_knapsack.evals import eval_custo, eval_juiz, eval_otimalidade, eval_resposta, eval_selecao, resumir_linhas
from rag_knapsack.gerador import Gerador, RespostaGerada
from rag_knapsack.juiz import Juiz, Veredito
from rag_knapsack.llm import ErroDeModelo
from rag_knapsack.seletores import Selecao, Seletor


@dataclass
class Resultado:
    exemplo: Exemplo
    selecao: Selecao
    latencia_selecao_ms: float
    resposta: RespostaGerada | None = None
    veredito: Veredito | None = None
    erro: str | None = None

    def como_dict(self, cfg: Config) -> dict:
        exemplo, selecao = self.exemplo, self.selecao
        linha = {
            "id": exemplo.id,
            "tipo": exemplo.tipo,
            "nivel": exemplo.nivel,
            "pergunta": exemplo.pergunta,
            "resposta_referencia": exemplo.resposta,
            "algoritmo": selecao.algoritmo,
            "orcamento": selecao.orcamento,
            "candidatos": [
                {"id": c.id, "tokens": c.tokens, "score": round(c.score, 6), "de_suporte": c.de_suporte}
                for c in exemplo.chunks
            ],
            "selecionados": [chunk.id for chunk in selecao.chunks],
            **eval_selecao(selecao, exemplo),
            **eval_otimalidade(selecao, exemplo),
            "latencia_selecao_ms": self.latencia_selecao_ms,
            "resposta_gerada": self.resposta.texto if self.resposta else None,
        }
        if self.resposta:
            linha |= eval_resposta(self.resposta.texto, exemplo) | eval_custo(self.resposta, cfg)
        if self.veredito:
            linha |= eval_juiz(self.veredito)
        linha["erro"] = self.erro
        return linha


def _cache_llm(cfg: Config, nome: str, usar_cache: bool) -> CacheJson:
    """Respostas do LLM guardadas em data/cache/, por modelo e reasoning_effort."""
    if not usar_cache:
        return CacheJson()
    return CacheJson(cfg.dir_dados / "cache" / (re.sub(r"[^A-Za-z0-9_.-]+", "_", nome) + ".json"))


def criar_gerador(cfg: Config, usar_cache: bool = True) -> Gerador:
    cache = _cache_llm(cfg, f"respostas-{cfg.modelo_gerador}-{cfg.esforco_gerador}", usar_cache)
    return Gerador(cfg.modelo_gerador, cfg.esforco_gerador, cache)


def criar_juiz(cfg: Config, usar_cache: bool = True) -> Juiz:
    cache = _cache_llm(cfg, f"vereditos-{cfg.modelo_juiz}-{cfg.esforco_juiz}", usar_cache)
    return Juiz(cfg.modelo_juiz, cfg.esforco_juiz, cache)


def selecionar(exemplos: Sequence[Exemplo], seletor: Seletor, orcamento: int) -> list[Resultado]:
    resultados = []
    for exemplo in exemplos:
        inicio = time.perf_counter()
        selecao = seletor.selecionar(exemplo.chunks, orcamento)
        resultados.append(Resultado(exemplo, selecao, (time.perf_counter() - inicio) * 1000))
    return resultados


def _uma_chamada_por_grupo(
    resultados: Sequence[Resultado],
    agrupar: Callable[[Resultado], Hashable],
    chamar: Callable[[Resultado], None],
    copiar: Callable[[Resultado, Resultado], None],
    workers: int,
) -> None:
    """Faz uma chamada por grupo de resultados equivalentes e copia o retorno para o resto.

    Quando dois algoritmos (ou dois orçamentos) escolhem o mesmo contexto para a mesma
    pergunta, o prompt é idêntico: chamar o LLM de novo só gastaria e acrescentaria ruído.
    """
    grupos: dict[Hashable, list[Resultado]] = defaultdict(list)
    for resultado in resultados:
        grupos[agrupar(resultado)].append(resultado)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda grupo: chamar(grupo[0]), grupos.values()))
    for grupo in grupos.values():
        for outro in grupo[1:]:
            copiar(grupo[0], outro)


def _com_nova_tentativa(chamada: Callable[[], object], tentativas: int = 2, espera_s: float = 2.0) -> object:
    """Repete uma chamada ao LLM que falhou (o agno não faz retry por padrão)."""
    for tentativa in range(tentativas):
        try:
            return chamada()
        except ErroDeModelo:
            if tentativa == tentativas - 1:
                raise
            time.sleep(espera_s)
    raise AssertionError("inalcançável")


def gerar_respostas(resultados: Sequence[Resultado], gerador: Gerador, workers: int = 4) -> None:
    def chamar(resultado: Resultado) -> None:
        try:
            resultado.resposta = _com_nova_tentativa(
                lambda: gerador.responder(resultado.exemplo.pergunta, resultado.selecao.chunks)
            )
        except ErroDeModelo as erro:
            resultado.erro = str(erro)

    def copiar(origem: Resultado, destino: Resultado) -> None:
        destino.resposta, destino.erro = origem.resposta, origem.erro

    _uma_chamada_por_grupo(
        resultados,
        agrupar=lambda r: (r.exemplo.id, tuple(chunk.id for chunk in r.selecao.chunks)),
        chamar=chamar,
        copiar=copiar,
        workers=workers,
    )


def julgar(resultados: Sequence[Resultado], juiz: Juiz, workers: int = 4) -> None:
    def chamar(resultado: Resultado) -> None:
        try:
            resultado.veredito = _com_nova_tentativa(
                lambda: juiz.avaliar(resultado.exemplo.pergunta, resultado.exemplo.resposta, resultado.resposta.texto)
            )
        except ErroDeModelo as erro:
            resultado.erro = str(erro)

    def copiar(origem: Resultado, destino: Resultado) -> None:
        destino.veredito = origem.veredito
        destino.erro = destino.erro or origem.erro

    respondidos = [resultado for resultado in resultados if resultado.resposta is not None]
    _uma_chamada_por_grupo(
        respondidos,
        agrupar=lambda r: (r.exemplo.id, r.resposta.texto),
        chamar=chamar,
        copiar=copiar,
        workers=workers,
    )


def resumir(resultados: Sequence[Resultado], cfg: Config) -> dict[str, float]:
    return resumir_linhas([resultado.como_dict(cfg) for resultado in resultados])


def gravar_jsonl(linhas: Sequence[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for linha in linhas:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")
