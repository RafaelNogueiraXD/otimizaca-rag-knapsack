"""Cache chave -> valor em JSON, para scores de embeddings e respostas do LLM."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


def chave(*partes: Any) -> str:
    return hashlib.sha256(json.dumps(partes, ensure_ascii=False).encode()).hexdigest()


class CacheJson:
    """Dicionário persistido em disco; sem caminho, fica só em memória.

    Reexecutar um experimento reaproveita o que já foi calculado (e pago), e algoritmos
    diferentes que montam o mesmo contexto recebem exatamente a mesma resposta.
    """

    def __init__(self, caminho: Path | None = None) -> None:
        self.caminho = caminho
        self._dados: dict[str, Any] = {}
        self._trava = threading.Lock()
        if caminho is not None and caminho.exists():
            self._dados = json.loads(caminho.read_text(encoding="utf-8"))

    def __contains__(self, chave_: str) -> bool:
        return chave_ in self._dados

    def get(self, chave_: str) -> Any:
        return self._dados.get(chave_)

    def set(self, chave_: str, valor: Any) -> None:
        with self._trava:
            self._dados[chave_] = valor

    def salvar(self) -> None:
        """Grava mesclando com o que já está no disco, para não apagar entradas de outro processo."""
        if self.caminho is None:
            return
        with self._trava:
            if self.caminho.exists():
                self._dados = json.loads(self.caminho.read_text(encoding="utf-8")) | self._dados
            conteudo = json.dumps(self._dados, ensure_ascii=False)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        temporario = self.caminho.with_suffix(".tmp")
        temporario.write_text(conteudo, encoding="utf-8")
        temporario.replace(self.caminho)
