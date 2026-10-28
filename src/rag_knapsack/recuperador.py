"""Recuperador: dá a cada chunk candidato seu valor v_i (score) e seu peso w_i (tokens)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from agno.knowledge.embedder.base import Embedder
from agno.knowledge.embedder.openai import OpenAIEmbedder

from rag_knapsack.cache import CacheJson
from rag_knapsack.config import Config
from rag_knapsack.dados import Exemplo
from rag_knapsack.tokens import ContadorTokens


def criar_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "openai":
        return OpenAIEmbedder(id=cfg.modelo_embedding)
    # Importa o onnxruntime, então só carregamos quando o embedder local for escolhido.
    from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
    from fastembed import TextEmbedding

    modelo = TextEmbedding(model_name=cfg.modelo_embedding, cache_dir=str(cfg.dir_dados / "cache" / "fastembed"))
    return FastEmbedEmbedder(id=cfg.modelo_embedding, fastembed_client=modelo)


def embutir(embedder: Embedder, textos: list[str]) -> np.ndarray:
    """Embute vários textos; na OpenAI vai tudo numa única requisição."""
    if isinstance(embedder, OpenAIEmbedder):
        resposta = embedder.client.embeddings.create(model=embedder.id, input=textos)
        vetores = [item.embedding for item in sorted(resposta.data, key=lambda item: item.index)]
    else:
        vetores = [embedder.get_embedding(texto) for texto in textos]
    return np.asarray(vetores, dtype=np.float64)


def similaridade_cosseno(consulta: np.ndarray, documentos: np.ndarray) -> np.ndarray:
    consulta = consulta / np.linalg.norm(consulta)
    documentos = documentos / np.linalg.norm(documentos, axis=1, keepdims=True)
    return documentos @ consulta


class Recuperador:
    """Etapa de recuperação: devolve os candidatos com valor e peso preenchidos.

    No cenário distractor os candidatos já vêm com a pergunta, então o recuperador só os
    pontua. Os scores ficam em cache no disco para que Top-K, Greedy e Knapsack recebam
    exatamente os mesmos valores, mesmo rodando em processos diferentes.
    """

    def __init__(self, embedder: Embedder, contador: ContadorTokens, arquivo_cache: Path | None = None) -> None:
        self.embedder = embedder
        self.contador = contador
        self._scores = CacheJson(arquivo_cache)

    @staticmethod
    def _chave(pergunta: str, bloco: str) -> str:
        return hashlib.sha256(f"{pergunta}\x00{bloco}".encode()).hexdigest()

    def preparar(self, exemplo: Exemplo) -> Exemplo:
        chaves = [self._chave(exemplo.pergunta, chunk.bloco) for chunk in exemplo.chunks]
        if any(chave not in self._scores for chave in chaves):
            vetores = embutir(self.embedder, [exemplo.pergunta, *(chunk.bloco for chunk in exemplo.chunks)])
            for chave, score in zip(chaves, similaridade_cosseno(vetores[0], vetores[1:]), strict=True):
                self._scores.set(chave, float(score))
        chunks = tuple(
            replace(chunk, tokens=self.contador.contar(chunk.bloco), score=self._scores.get(chave))
            for chunk, chave in zip(exemplo.chunks, chaves, strict=True)
        )
        return exemplo.com_chunks(chunks)

    def salvar_cache(self) -> None:
        self._scores.salvar()


def criar_recuperador(cfg: Config) -> Recuperador:
    nome_cache = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"scores-{cfg.embedder}-{cfg.modelo_embedding}") + ".json"
    return Recuperador(criar_embedder(cfg), ContadorTokens(cfg.modelo_gerador), cfg.dir_dados / "cache" / nome_cache)
