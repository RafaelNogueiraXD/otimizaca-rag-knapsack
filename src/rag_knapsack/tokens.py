"""Contagem de tokens: o peso w_i de cada chunk na mochila."""

from __future__ import annotations

import tiktoken

# O tiktoken 0.14 ainda não mapeia o gpt-6-luna. Usamos o o200k_base (GPT-4o e GPT-5) como
# aproximação; o custo real de cada chamada vem do `usage` da API e é registrado nos resultados.
CODIFICACAO_PADRAO = "o200k_base"


class ContadorTokens:
    def __init__(self, modelo: str) -> None:
        try:
            self.codificacao = tiktoken.encoding_for_model(modelo)
        except KeyError:
            self.codificacao = tiktoken.get_encoding(CODIFICACAO_PADRAO)

    @property
    def nome(self) -> str:
        return self.codificacao.name

    def contar(self, texto: str) -> int:
        return len(self.codificacao.encode(texto))
