"""Juiz: Agent-as-a-judge do agno (`AgentAsJudgeEval`, modo binário).

EM e F1 penalizam respostas certas escritas de outro jeito ("JFK" contra "John F. Kennedy").
O juiz decide se a resposta gerada é equivalente ao gabarito, e justifica a decisão.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agno.eval.agent_as_judge import AgentAsJudgeEval
from agno.metrics import RunMetrics

from rag_knapsack.cache import CacheJson, chave
from rag_knapsack.llm import ErroDeModelo, Gasto, modelo_openai

CRITERIOS = (
    "The input contains a question and its reference answer; the output is the answer produced by a "
    "question-answering system. The output PASSES if it gives the same answer as the reference, even if "
    "worded differently (e.g. 'JFK' for 'John F. Kennedy', 'in 1990' for '1990', 'Yes.' for 'yes') or with "
    "harmless extra detail. It FAILS if it names a different entity, number, date or yes/no value, answers "
    "'unknown', or hedges between several candidates."
)
DIRETRIZES = [
    "Compare the output only against the reference answer; do not use your own knowledge to decide what is true.",
    "Ignore differences in capitalization, punctuation and articles.",
]


@dataclass(frozen=True)
class Veredito:
    correto: bool
    justificativa: str


class Juiz:
    def __init__(self, modelo: str, esforco: str, cache: CacheJson | None = None) -> None:
        self.modelo = modelo
        self.esforco = esforco
        self.cache = cache
        self.gasto = Gasto()

    def avaliar(self, pergunta: str, referencia: str, resposta: str) -> Veredito:
        chave_cache = chave(self.modelo, self.esforco, CRITERIOS, DIRETRIZES, pergunta, referencia, resposta)
        if self.cache is not None and chave_cache in self.cache:
            return Veredito(**self.cache.get(chave_cache))
        # Uma avaliação por chamada, pelo mesmo motivo do gerador: as chamadas rodam em paralelo.
        avaliacao = AgentAsJudgeEval(
            name="hotpotqa-equivalencia",
            criteria=CRITERIOS,
            scoring_strategy="binary",
            additional_guidelines=DIRETRIZES,
            model=modelo_openai(self.modelo, self.esforco),
            show_spinner=False,
            telemetry=False,
        )
        metricas = RunMetrics()
        resultado = avaliacao.run(
            input=f"Question: {pergunta}\nReference answer: {referencia}", output=resposta, run_metrics=metricas
        )
        self.gasto.registrar(metricas.input_tokens, metricas.output_tokens)
        # O agno registra a falha no log e devolve um resultado vazio em vez de levantar exceção.
        if resultado is None or not resultado.results:
            raise ErroDeModelo(f"O juiz ({self.modelo}) não devolveu veredito")
        veredito = Veredito(correto=resultado.results[0].passed, justificativa=resultado.results[0].reason)
        if self.cache is not None:
            self.cache.set(chave_cache, asdict(veredito))
        return veredito
