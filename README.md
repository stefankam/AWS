# Availability-Aware Collaborative Financial SLM Adaptation

This project creates a realistic experimental environment for studying:

- decentralized Small Language Model (SLM) adaptation,
- intermittent client participation,
- temporal semantic drift,
- personalized financial assistants,
- and availability-aware federated scheduling.

The pipeline transforms public financial language datasets into a simulated collaborative edge-learning environment where:

- clients represent personalized financial assistants,
- semantic distributions evolve over time,
- clients participate intermittently,
- and correlated availability reshapes collaborative adaptation.

The generated dataset can be used for:
- federated learning,
- continual SLM adaptation,
- availability-aware scheduling,
- fairness-aware optimization,
- and distributed edge-learning experiments.

---

# Pipeline Overview

The system consists of four major stages:

1. Base Financial Language Dataset  
2. Convert Into Personal Clients  
3. Add Temporal Drift  
4. Add Realistic Availability  

---

# 1. Base Financial Language Dataset

The first stage loads a public financial language dataset and converts it into a standardized format suitable for distributed SLM adaptation.

## Purpose

The goal is not to prove that centralized LLMs have never seen financial data. Instead, the objective is to model:

- evolving financial semantics,
- temporally changing terminology,
- localized market narratives,
- and personalized adaptation behavior.

Financial domains are particularly suitable because:

- terminology evolves rapidly,
- market sentiment changes continuously,
- new entities emerge frequently,
- and semantic drift occurs naturally over time.

This makes collaborative edge-side adaptation meaningful.

---

## Supported Datasets

The pipeline supports HuggingFace datasets such as:

- FiQA
- FinGPT
- Financial News datasets
- Financial sentiment datasets
- custom CSV datasets

Example:

```python
dataset_name = "ChanceFocus/fiqa-sentiment-classification"
```

---

# 2. Convert Into Personal Clients

The second stage converts centralized financial data into decentralized personalized edge clients.

## Motivation

In realistic decentralized systems:

- clients are not passive mirrors of public news,
- clients represent personalized semantic environments,
- each client adapts to user-specific interests,
- and no centralized server fully observes all local distributions.

The goal is to simulate:

- personal financial assistants,
- mobile finance copilots,
- retail investors,
- crypto traders,
- regional market users,
- institutional analysts,
- and personalized financial interaction patterns.

---

## Client Personas

The system creates synthetic user personas such as:

| Persona | Example Interests |
|---|---|
| retail_etf | ETFs, dividends, retirement |
| crypto_trader | crypto, DeFi, tokens |
| eu_investor | ECB, European markets |
| asia_market | Nikkei, China markets |
| macro_analyst | inflation, GDP, rates |
| stock_picker | earnings, valuation |

---

# 3. Add Temporal Drift

The third stage introduces evolving semantic distributions over time.

## Motivation

Financial language is highly nonstationary.

Examples include:
- emerging companies,
- regulatory changes,
- geopolitical events,
- evolving market narratives,
- and changing financial terminology.

Temporal drift enables the study of:
- continual adaptation,
- delayed semantic propagation,
- and adaptation lag.

---

## Temporal Rounds

The dataset is chronologically partitioned into federated learning rounds.

Example:

| Round | Time Window |
|---|---|
| 0 | earliest data |
| 10 | later semantic state |
| 49 | newest semantic state |

---

# 4. Add Realistic Availability

The final stage simulates intermittent and correlated edge participation.

## Motivation

In realistic mobile-edge systems:

- devices disconnect,
- users sleep,
- phones charge,
- mobile connectivity fluctuates,
- and participation naturally follows regional patterns.

Unlike stable datacenter infrastructure, edge-side personalized assistants are inherently intermittent.

---

## Correlated Availability

Clients are grouped by:
- region,
- timezone,
- and behavioral patterns.

Examples:

| Group | Active Period |
|---|---|
| ASIA | Asian daytime |
| EU | European daytime |
| US | US trading hours |
| GLOBAL | continuously active |

This creates realistic synchronized participation patterns.

---

# Research Goals

This framework enables research on:

- collaborative SLM adaptation,
- edge-side continual learning,
- intermittent participation,
- fairness-aware FL,
- semantic drift,
- decentralized personalization,
- and availability-aware scheduling.
