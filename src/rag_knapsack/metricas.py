"""Métricas de avaliação.

- EM e F1 da resposta seguem o script oficial do HotpotQA (hotpot_evaluate_v1.py).
- As métricas de suporte medem, sem LLM, se a seleção trouxe os parágrafos que
  sustentam a resposta (em multi-hop são necessários os dois).
"""

from __future__ import annotations

import re
import string
from collections import Counter

from rag_knapsack.dados import Exemplo
from rag_knapsack.seletores import Selecao

RESPOSTA_DESCONHECIDA = "unknown"  # o gerador responde isso quando o contexto não basta
_PONTUACAO = set(string.punctuation)


def normalizar_resposta(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(caractere for caractere in texto if caractere not in _PONTUACAO)
    texto = re.sub(r"\b(a|an|the)\b", " ", texto)
    return " ".join(texto.split())


def exact_match(predita: str, referencia: str) -> float:
    return float(normalizar_resposta(predita) == normalizar_resposta(referencia))


def f1(predita: str, referencia: str) -> float:
    predita, referencia = normalizar_resposta(predita), normalizar_resposta(referencia)
    # Para sim/não, acertar parcialmente não existe: ou é igual, ou F1 = 0.
    if (predita in {"yes", "no", "noanswer"} or referencia in {"yes", "no", "noanswer"}) and predita != referencia:
        return 0.0
    tokens_predita, tokens_referencia = predita.split(), referencia.split()
    em_comum = sum((Counter(tokens_predita) & Counter(tokens_referencia)).values())
    if em_comum == 0:
        return 0.0
    precisao = em_comum / len(tokens_predita)
    revocacao = em_comum / len(tokens_referencia)
    return 2 * precisao * revocacao / (precisao + revocacao)


def sem_resposta(predita: str) -> bool:
    return normalizar_resposta(predita) == RESPOSTA_DESCONHECIDA


def recall_suporte(selecao: Selecao, exemplo: Exemplo) -> float:
    """Fração dos parágrafos de suporte que entraram no contexto."""
    selecionados = {chunk.id for chunk in selecao.chunks}
    return len(selecionados & exemplo.ids_suporte) / len(exemplo.ids_suporte)


def precisao_suporte(selecao: Selecao, exemplo: Exemplo) -> float:
    """Fração do contexto escolhido que é parágrafo de suporte."""
    if not selecao.chunks:
        return 0.0
    return sum(chunk.id in exemplo.ids_suporte for chunk in selecao.chunks) / len(selecao.chunks)


def suporte_completo(selecao: Selecao, exemplo: Exemplo) -> bool:
    """Todos os parágrafos de suporte estão no contexto (condição para o multi-hop)."""
    return exemplo.ids_suporte <= {chunk.id for chunk in selecao.chunks}
