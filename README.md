# P2-ETF-NEURAL-OPERATOR

**Fourier Neural Operator for Cross‑Asset Exchange Option Pricing and ETF Ranking**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-NEURAL-OPERATOR/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-NEURAL-OPERATOR/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--neural--operator--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-neural-operator-results)

## Overview

`P2-ETF-NEURAL-OPERATOR` learns the mapping from covariance surfaces to Margrabe exchange option prices using a **Fourier Neural Operator (FNO)**. This function‑to‑function learning paradigm captures complex cross‑asset relationships that traditional models miss. ETFs are ranked by their average exchange option price (higher is better).

## Methodology

1. **Covariance Surface Generation**: Rolling 63‑day covariance matrices from 2008–2026 YTD.
2. **Margrabe Target**: Closed‑form exchange option prices for all ETF pairs.
3. **FNO Training**: Learns the operator `G: Covariance → Pairwise Prices`.
4. **ETF Ranking**: Average price when exchanging the ETF for any other.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)
