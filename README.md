# RAG com Knapsack: Otimização de Contexto para Agentes de IA

**Autores:** Eduardo Leal de Carvalho, Rafael Nogueira Rodrigues, Yang da Fontoura Rodrigues

Repositório oficial do projeto de pesquisa que propõe o uso do **Problema da Mochila (Knapsack Problem)** como estratégia de otimização da seleção de contexto em sistemas de **Retrieval-Augmented Generation (RAG)** para agentes de Inteligência Artificial.

## Resumo

Sistemas RAG recuperam trechos de documentos relevantes para enriquecer o contexto fornecido a Modelos de Linguagem de Grande Escala (LLMs). Entretanto, a seleção ingênua dos trechos mais similares (top-k) nem sempre produz o melhor conjunto de contexto, pois ignora restrições como o limite de tokens da janela de contexto e a necessidade de maximizar a relevância total do conjunto selecionado. Este trabalho propõe modelar esse problema como uma instância do **Problema da Mochila 0/1**, em que cada trecho recuperado possui um "peso" (quantidade de tokens) e um "valor" (score de relevância/similaridade), buscando maximizar a relevância total do contexto respeitando o limite de tokens disponível.

## 1. Motivação

A motivação central do projeto é aprimorar a qualidade do contexto fornecido a agentes de IA baseados em RAG, tratando a seleção de trechos como um problema de otimização combinatória, em vez de uma simples ordenação por similaridade (top-k).

## 2. Formulação do Problema

O problema de selecionar os melhores trechos de contexto para um LLM é modelado como o **Problema da Mochila 0/1**:

- **Itens:** trechos de texto (chunks) recuperados pelo sistema de RAG.
- **Peso:** número de tokens de cada trecho.
- **Valor:** score de relevância/similaridade do trecho em relação à consulta (query).
- **Capacidade da mochila:** limite máximo de tokens da janela de contexto do LLM.
- **Objetivo:** selecionar o subconjunto de trechos que maximize a soma dos valores (relevância) sem exceder a capacidade (limite de tokens).

## 3. Arquitetura do Sistema

O sistema é composto pelas seguintes etapas:

1. Recebimento da consulta do usuário (query).
2. Recuperação de candidatos (chunks) via busca vetorial/semântica.
3. Cálculo de peso (tokens) e valor (score de relevância) para cada candidato.
4. Seleção do subconjunto ótimo de trechos através do algoritmo de otimização (Knapsack).
5. Montagem do contexto final e envio ao LLM para geração da resposta.

## 4. Algoritmos Comparados

Para avaliar os ganhos da abordagem proposta, são comparados os seguintes algoritmos de seleção de contexto:

| Algoritmo | Descrição | Complexidade | Garante Ótimo |
|---|---|---|---|
| **Top-k** | Seleciona os k trechos mais similares à query, ordenados por score, até atingir o limite de tokens ou a quantidade k. | O(n log n) | Não |
| **Greedy (Guloso)** | Seleciona trechos ordenados pela razão valor/peso (relevância por token), de forma gulosa, até esgotar a capacidade. | O(n log n) | Não (aproximado) |
| **Knapsack (Programação Dinâmica)** | Resolve o problema da mochila 0/1 de forma exata (ou com aproximação via PD), maximizando a relevância total sujeita ao limite de tokens. | O(n·W) | Sim (exato) |

## 5. Metodologia Experimental

A metodologia experimental consiste em:

- Definição de um conjunto de documentos e consultas de teste.
- Execução dos três algoritmos de seleção de contexto (Top-k, Greedy e Knapsack) sobre os mesmos candidatos recuperados.
- Medição de métricas como relevância total do contexto selecionado, uso da capacidade de tokens e qualidade da resposta final gerada pelo LLM.
- Comparação dos resultados entre as três abordagens.

## 6. Resultados Esperados e Hipóteses

Espera-se que a abordagem baseada em Knapsack (Programação Dinâmica) produza um contexto com maior relevância total agregada em comparação às abordagens Top-k e Greedy, aproveitando melhor a capacidade de tokens disponível e, consequentemente, melhorando a qualidade das respostas geradas pelo agente de IA.

## 7. Cronograma

Serão definidas etapas de implementação, experimentação e análise de resultados ao longo do desenvolvimento do projeto, com entregas organizadas por algoritmo (Top-k, Greedy e Knapsack).

## 8. Referências

Referências bibliográficas utilizadas no desenvolvimento deste trabalho encontram-se detalhadas no artigo acadêmico associado a este repositório.

---

## Implementação (Entrega 4)

Os três algoritmos da proposta estão implementados, testados e comparados: **Top-K**, **Greedy** e **Knapsack DP**. Também há a variante **Top-K com corte**, que enche o orçamento T por ordem de score, sem limite de k, e serve de base de comparação justa. Há ainda funções de avaliação (evals), um experimento comparativo e um painel em Streamlit.

Tecnologias: Python 3.12 com [uv](https://docs.astral.sh/uv/), [agno](https://docs.agno.com) (agentes, `AgentAsJudgeEval` e `PerformanceEval`), modelos da OpenAI (`gpt-6-luna` como gerador e juiz, `text-embedding-3-small` para os embeddings), [taskipy](https://github.com/taskipy/taskipy) para as tasks e [Streamlit](https://streamlit.io) para o painel.

### Como rodar

```bash
uv sync
cp .env.example .env   # preencha OPENAI_API_KEY
uv run task dados      # baixa o HotpotQA dev distractor (7.405 perguntas, ~45 MB)
uv run task avaliar    # compara os algoritmos: 30 perguntas, T = 256, 512 e 1024
uv run task painel     # abre o painel com os resultados
```

| Task | O que faz |
|---|---|
| `uv run task avaliar` | Experimento comparativo: todos os algoritmos nas mesmas perguntas e orçamentos, com todos os evals |
| `uv run task painel` | Painel Streamlit com gráficos e a comparação pergunta a pergunta |
| `uv run task topk` / `greedy` / `knapsack` | Um algoritmo só (aceita `--n`, `--orcamento`, `--sem-llm` etc.) |
| `uv run task dados` | Baixa o dataset para `data/` (retoma se for interrompido) |
| `uv run task amostra` | Regera a amostra fixa de 10 perguntas usada nos testes |
| `uv run task test` | Roda os testes (adicione `-s` para ver as métricas) |

Opções do `avaliar`: `--n 100`, `--orcamentos 128 256 512 1024`, `--algoritmos knapsack greedy`, `--sem-llm` (só seleção, otimalidade e desempenho), `--sem-juiz`, `--sem-cache` e `--workers 8`. Cada execução grava uma pasta `resultados/avaliacao-<data>/`, que o painel lê.

### Algoritmos (`src/rag_knapsack/seletores.py`)

| Algoritmo | Estratégia | Complexidade | Respeita T? | Ótimo da soma de scores? |
|---|---|---|---|---|
| Top-K | os k = 4 chunks de maior score | O(n log n) | não | não |
| Top-K com corte | maior score primeiro até encher T, sem limite de k | O(n log n) | sim | não |
| Greedy | maior score/token primeiro, pulando o que não cabe | O(n log n) | sim | não; perde no máximo o valor de um chunk |
| Knapsack DP | mochila 0/1 exata, com reconstrução da solução | O(n·T) | sim | sim |

Todos entregam os chunks escolhidos na mesma ordem (score decrescente; em empate, a ordem original dos candidatos), para que só o *conjunto* mude entre algoritmos e não a posição no prompt (*Lost in the Middle*). Chunks com score ≤ 0 nunca entram na mochila. O Knapsack foi conferido contra força bruta em 400 instâncias aleatórias e em todas as perguntas da amostra.

A tabela da proposta diz que o Greedy é "1/2-ótimo". Isso só vale para o guloso *modificado*, que compara a solução gulosa com o melhor chunk isolado; o Greedy implementado é o da proposta, sem essa garantia.

### Evals (`src/rag_knapsack/evals.py`)

| Função | O que mede | Hipótese |
|---|---|---|
| `eval_selecao` | tokens usados, ocupação e estouro de T, recall e precisão dos parágrafos de suporte | |
| `eval_otimalidade` | valor da seleção ÷ valor ótimo da mochila no mesmo T | H2 |
| `eval_resposta` | EM e F1 oficiais do HotpotQA, e respostas "unknown" | H1 |
| `eval_juiz` | veredito do Agent-as-a-judge (`AgentAsJudgeEval` do agno, modo binário) | H1 |
| `eval_custo` | tokens cobrados pela API, custo e tempo de geração | custo |
| `eval_desempenho` | tempo e pico de memória de uma seleção (`PerformanceEval` do agno) | H3 |

Uma falha da API conta como resposta errada (todos os algoritmos ficam com o mesmo número de perguntas) e cada chamada que falha é repetida uma vez. Se ainda sobrar erro, o painel avisa; rodar o mesmo `avaliar` de novo refaz só as falhas.

O experimento faz **uma chamada ao LLM por contexto distinto**: quando dois algoritmos escolhem o mesmo conjunto de chunks para a mesma pergunta, a resposta é reaproveitada. Isso economiza e evita que a comparação seja contaminada por variação do modelo. Respostas e vereditos ficam em cache em `data/cache/`, então reexecutar não gasta de novo.

### Painel (`uv run task painel`)

- **Visão geral:** qualidade da resposta (juiz, F1, EM), contexto selecionado (recall, suporte completo, estouro) e tamanho do contexto contra T.
- **Orçamento × qualidade:** cada métrica em função de T, para ver custo contra qualidade.
- **Otimalidade e latência:** distância do ótimo da mochila (H2) e tempo de seleção contra tempo do LLM (H3).
- **Perguntas:** para cada pergunta, a resposta ideal (gabarito e parágrafos de suporte) e o que cada algoritmo respondeu, com o veredito do juiz, os chunks escolhidos e as justificativas. Há filtros por divergência e por erro.
- **Exemplo de Brasília:** o exemplo da seção 8, interativo (T e k ajustáveis).

### Testes

`uv run task test` roda 45 testes:
- **Seletores:** exemplo de Brasília, Greedy que pula o que não cabe, empates, casos de borda e força bruta.
- **Evals:** EM/F1 do script oficial, otimalidade, falhas contando como erro, deduplicação de chamadas e formato do experimento.
- **Painel:** todas as abas renderizam (via `AppTest`).
- **HotpotQA:** para cada algoritmo, os três testes (seleção, EM/F1 e Agent-as-a-judge). O de seleção não chama o LLM, só os embeddings (com `EMBEDDER=openai` ele precisa da chave). Os de EM/F1 e do juiz chamam a API, são marcados com `llm` e são pulados sem `OPENAI_API_KEY`.

### Primeiros resultados

Estes resultados vêm de 40 perguntas do dev, com k = 4, `gpt-6-luna` (raciocínio `medium`) e `text-embedding-3-small`. Gerar tudo custou cerca de US$ 0,03 em chamadas à API.

| T | Algoritmo | Juiz | F1 | Os 2 de suporte | Estouro de T | Tokens | Valor ÷ ótimo |
|---|---|---|---|---|---|---|---|
| 256 | Top-K | 0,60 | 0,62 | 0,70 | 100% | 442 | — |
| 256 | Greedy | 0,47 | 0,48 | 0,47 | 0% | 213 | 0,97 |
| 256 | Knapsack DP | 0,57 | 0,55 | 0,57 | 0% | 233 | 1,00 |
| 256 | Top-K com corte | 0,57 | 0,56 | 0,53 | 0% | 226 | 0,92 |
| 512 | Top-K | 0,60 | 0,62 | 0,70 | 25% | 442 | 0,82 (30/40) |
| 512 | Greedy | 0,72 | 0,69 | 0,75 | 0% | 454 | 0,98 |
| 512 | Knapsack DP | 0,68 | 0,68 | 0,78 | 0% | 481 | 1,00 |
| 512 | Top-K com corte | 0,70 | 0,70 | 0,78 | 0% | 473 | 0,94 |
| 1024 | Top-K | 0,60 | 0,62 | 0,70 | 0% | 442 | 0,59 |
| 1024 | Greedy | 0,72 | 0,67 | 0,97 | 0% | 926 | 1,00 |
| 1024 | Knapsack DP | 0,72 | 0,66 | 0,93 | 0% | 940 | 1,00 |
| 1024 | Top-K com corte | 0,70 | 0,65 | 0,88 | 0% | 939 | 0,97 |

Como ler a tabela:
- "Valor ÷ ótimo" é a média só das seleções que respeitam T. Para o Top-K em T = 512, são 30 das 40 perguntas; em T = 256 todas estouram, por isso "—".
- Com 40 perguntas, cada pergunta vale 2,5 pontos percentuais. Entre Greedy, Knapsack e Top-K com corte, em T = 512 e 1024, as diferenças são de 1 ou 2 perguntas: **ainda não dá para afirmar a H1**. Para isso, rode com `--n 300` ou mais (estimativa: menos de US$ 0,30) e use um teste pareado.

O que esses números já mostram:
- O Top-K "padrão" (k = 4) não respeita o orçamento: estoura T em todas as perguntas com T = 256. Com T = 1024, ele usa só 442 tokens e fica atrás dos que enchem o orçamento.
- O Greedy é o mais fraco com orçamento apertado (T = 256): 47% de acerto contra 57% do Knapsack e do Top-K com corte.
- A mochila sempre atinge o ótimo da **soma dos scores**, mas isso não vira automaticamente mais acertos: em T = 512, o Top-K com corte tem 94% desse valor e o mesmo acerto (dentro do ruído). É o "o que estamos maximizando?" na prática: a soma de similaridades é uma proxy da qualidade do contexto.
- Trocar o embedder pelo local (`bge-small`), nas mesmas 40 perguntas, muda pouco: em T = 256, os dois parágrafos de suporte entram em 53% das perguntas com o Knapsack e 57% com o Top-K com corte (com a OpenAI, 57% e 53%), uma diferença de 1 ou 2 perguntas.
- A seleção é desprezível perto da geração (H3): a DP leva 0,07 ms com T = 256 e 0,5 ms com T = 1024 (e de 11 a 43 KiB de memória, crescendo com T, como esperado de O(n·T)); uma chamada ao LLM leva cerca de 1,5 s.

### Decisões e limitações

- **Orçamento padrão T = 512:** os 10 candidatos somam em média ~1.300 tokens. Com T = 4.096 caberia tudo, e os algoritmos escolheriam o mesmo conjunto.
- **Peso em tokens:** o tiktoken ainda não conhece o `gpt-6-luna`, então usamos o `o200k_base` como aproximação. O custo real vem do `usage` da API.
- **Juiz:** por padrão é o mesmo modelo do gerador, o que pode favorecer as respostas dele. Dá para trocar com `MODELO_JUIZ` no `.env`.
- **Sem chave da OpenAI:** `EMBEDDER=fastembed` com `--sem-llm` roda só a seleção, a otimalidade e o desempenho.

Dados: HotpotQA (Yang et al., 2018), licença CC BY-SA 4.0. A amostra em `tests/fixtures/` é um recorte do conjunto dev.

## Estrutura de Branches

Este repositório utiliza branches dedicadas para o desenvolvimento independente de cada estratégia de seleção de contexto:

- `top-k`: implementação da abordagem de seleção por similaridade (Top-k).
- `greedy`: implementação da abordagem gulosa (Greedy) baseada na razão valor/peso.
- `knapsack`: implementação da abordagem exata via Problema da Mochila (Programação Dinâmica).

## Contato

- eduardocarvalho.aluno@unipampa.edu.br
- rafaelnogueira.aluno@unipampa.edu.br
- yangrodrigues.aluno@unipampa.edu.br
