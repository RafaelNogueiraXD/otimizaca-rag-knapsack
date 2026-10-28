"""Acesso aos modelos da OpenAI pelo agno."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agno.models.openai import OpenAIResponses


class ErroDeModelo(RuntimeError):
    """Falha numa chamada ao LLM.

    O `Agent.run` do agno não levanta exceção: devolve a mensagem de erro como conteúdo da
    resposta, com status ERROR. Convertê-la nesta exceção evita que um erro de API seja
    avaliado como se fosse a resposta do modelo.
    """


def modelo_openai(id_modelo: str, esforco: str) -> OpenAIResponses:
    """Modelo pela Responses API, recomendada pela OpenAI para o gpt-6-luna.

    `store=False` faz a OpenAI não guardar as respostas; cada chamada é independente.
    Não enviamos `temperature`: nos modelos de raciocínio quem controla o comportamento
    é o `reasoning_effort`.
    """
    return OpenAIResponses(id=id_modelo, reasoning_effort=esforco, store=False)


@dataclass
class Gasto:
    """Tokens das chamadas que de fato foram à API (respostas vindas do cache não contam)."""

    chamadas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    _trava: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def registrar(self, tokens_entrada: int, tokens_saida: int) -> None:
        with self._trava:
            self.chamadas += 1
            self.tokens_entrada += tokens_entrada
            self.tokens_saida += tokens_saida

    def custo_usd(self, preco_entrada_mtok: float, preco_saida_mtok: float) -> float:
        return (self.tokens_entrada * preco_entrada_mtok + self.tokens_saida * preco_saida_mtok) / 1e6
