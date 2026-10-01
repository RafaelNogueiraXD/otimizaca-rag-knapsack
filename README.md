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

## Estrutura de Branches

Este repositório utiliza branches dedicadas para o desenvolvimento independente de cada estratégia de seleção de contexto:

- `top-k`: implementação da abordagem de seleção por similaridade (Top-k).
- `greedy`: implementação da abordagem gulosa (Greedy) baseada na razão valor/peso.
- `knapsack`: implementação da abordagem exata via Problema da Mochila (Programação Dinâmica).

## Contato

- eduardocarvalho.aluno@unipampa.edu.br
- rafaelnogueira.aluno@unipampa.edu.br
- yangrodrigues.aluno@unipampa.edu.br
