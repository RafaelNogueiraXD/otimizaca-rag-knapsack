"""Configuração do experimento, lida de variáveis de ambiente (ou do arquivo `.env`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]

load_dotenv(RAIZ / ".env")
# Por padrão o agno envia telemetria (metadados de cada execução) para os servidores dele.
os.environ.setdefault("AGNO_TELEMETRY", "false")

MODELOS_EMBEDDING_PADRAO = {
    "openai": "text-embedding-3-small",
    "fastembed": "BAAI/bge-small-en-v1.5",  # local (ONNX), sem custo e sem chave de API
}


def _texto(nome: str, padrao: str) -> str:
    return os.getenv(nome, "").strip() or padrao


@dataclass(frozen=True)
class Config:
    modelo_gerador: str
    modelo_juiz: str
    esforco_gerador: str  # reasoning_effort: none | low | medium | high | xhigh | max
    esforco_juiz: str
    embedder: str  # openai | fastembed
    modelo_embedding: str
    top_k: int
    orcamento_tokens: int  # capacidade T da mochila (tokens de contexto)
    preco_entrada_mtok: float  # US$ por 1M de tokens de entrada do gerador
    preco_saida_mtok: float  # US$ por 1M de tokens de saída (inclui os de raciocínio)
    preco_entrada_juiz_mtok: float  # preços do juiz (padrão: os do gerador)
    preco_saida_juiz_mtok: float
    dir_dados: Path
    dir_resultados: Path

    @classmethod
    def do_ambiente(cls) -> Config:
        embedder = _texto("EMBEDDER", "openai").lower()
        if embedder not in MODELOS_EMBEDDING_PADRAO:
            raise ValueError(f"EMBEDDER deve ser um de {sorted(MODELOS_EMBEDDING_PADRAO)}, não {embedder!r}")
        preco_entrada = float(_texto("PRECO_ENTRADA_MTOK", "0.10"))
        preco_saida = float(_texto("PRECO_SAIDA_MTOK", "0.50"))
        return cls(
            modelo_gerador=_texto("MODELO_GERADOR", "gpt-6-luna"),
            modelo_juiz=_texto("MODELO_JUIZ", "gpt-6-luna"),
            esforco_gerador=_texto("ESFORCO_GERADOR", "medium"),
            esforco_juiz=_texto("ESFORCO_JUIZ", "low"),
            embedder=embedder,
            modelo_embedding=_texto("MODELO_EMBEDDING", MODELOS_EMBEDDING_PADRAO[embedder]),
            top_k=int(_texto("TOP_K", "4")),
            orcamento_tokens=int(_texto("ORCAMENTO_TOKENS", "512")),
            preco_entrada_mtok=preco_entrada,
            preco_saida_mtok=preco_saida,
            preco_entrada_juiz_mtok=float(_texto("PRECO_ENTRADA_JUIZ_MTOK", str(preco_entrada))),
            preco_saida_juiz_mtok=float(_texto("PRECO_SAIDA_JUIZ_MTOK", str(preco_saida))),
            dir_dados=RAIZ / "data",
            dir_resultados=RAIZ / "resultados",
        )

    @property
    def precisa_chave_openai(self) -> bool:
        """Os embeddings precisam da API quando o embedder é o da OpenAI."""
        return self.embedder == "openai"


def inteiro_positivo(texto: str) -> int:
    """Tipo de argparse para valores que precisam ser >= 1 (orçamento, n, k...)."""
    valor = int(texto)
    if valor < 1:
        raise ValueError(f"precisa ser >= 1: {texto}")
    return valor


def tem_chave_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())
