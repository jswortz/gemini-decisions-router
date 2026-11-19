# Gemini Decisions Router: Live Benchmarks

![Benchmark Dashboard](../assets/benchmark-dashboard.png)

## Executive Summary

We evaluated **Gemini Decisions Router** (`decisions_engine.py`) live against `https://generativelanguage.googleapis.com/v1beta` across **100 stratified developer and agent prompts** (`benchmarks/prompts_100.json`) spanning 6 engineering categories, plus **15 stratified end-to-end (E2E) model completions** across `flash_lite` (`gemini-3.1-flash-lite-preview`), `flash` (`gemini-2.5-flash`), and `pro` (`gemini-2.5-pro`).

| Headline Metric | Gemini Decisions Router (Tier 0+1) | Pure `gemini-decision-1` (Tier 1 Only) | `GenerateContent` JSON Router (`gemini-2.5-flash`) | Offline Lexical Baseline | Always-Pro Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample Size** | `N=100` | `N=100` | `N=15` (stratified) | `N=100` | `N=100` |
| **Router Latency `p50`** | **`278.2 ms`** *(Tier 0: `<0.1 ms`)* | **`235.3 ms`** | `1,124.3 ms` (**`4.0x` to `4.8x` slower**) | `0.01 ms` | `0.0 ms` |
| **Router Latency `p90` / `p99`** | `411.4 ms` / `1,146.2 ms` | **`374.2 ms` / `407.2 ms`** | `1,643.8 ms` / `2,112.3 ms` | `0.01 ms` / `0.01 ms` | `0.0 ms` |
| **Mean Router Output Tokens** | **`0.0` tokens** | **`0.0` tokens** | `144.3` tokens (`15.3` JSON + `128.9` thought) | `0.0` tokens | `0.0` tokens |
| **Exact 3-Tier Accuracy** | **`96.0%`** (`96/100`) | **`96.0%`** (`96/100`) | `100.0%` (`15/15`) | `79.0%` (`79/100`) | `30.0%` (`30/100`) |
| **5-Way `task_class` Accuracy** | **`93.3%`** (`84/90`) | **`93.3%`** (`84/90`) | `100.0%` (`15/15`) | `44.4%` (`40/90`) | `22.2%` (`20/90`) |
| **Critical Pro-Miss Rate** | **`0.0%`** (`0/30` missed) | **`0.0%`** (`0/30` missed) | `0.0%` (`0/5` missed) | **`36.7%`** (`11/30` missed) | `0.0%` (`0/30`) |
| **Non-Pro Over-Routed to Pro** | **`0.0%`** (`0/70`) | **`0.0%`** (`0/70`) | `0.0%` (`0/10`) | `2.9%` (`2/70`) | **`100.0%`** (`70/70` = `70%` of workload) |
| **100-Prompt Router Cost** | **`$0.0046`** | **`$0.0051`** | `$0.0417` (**`9.1x` higher**) | `$0.0000` | `$0.0000` |
| **100-Prompt Total E2E Time** | **`335.5 s` (`-29.1%`)** | `331.3 s` (`-30.0%`) | `416.0 s` (`-12.1%`) | N/A *(36.7% Pro-miss)* | `473.2 s` *(Baseline)* |
| **100-Prompt Total E2E Cost** | **`$0.225` (`-56.3%`)** | `$0.226` (`-56.2%`) | `$0.258` (`-50.0%`) | N/A | `$0.515` *(Baseline)* |

### Key Takeaways

1. **Single-Pass Prefill Eliminates Router Decoding Overhead:** Traditional `GenerateContent` JSON routing (`gemini-2.5-flash` with `responseMimeType: "application/json"`) required **`1,124.3 ms` at `p50`** (`1,643.8 ms` at `p90`) and generated an average of **`144.3` billed output + thinking tokens** just to emit a 15-token JSON string. By contrast, `gemini-decision-1` evaluates 3 structured questions (`choice`, `noul`, `score`) simultaneously in **one prefill pass with `0` output tokens**, achieving **`235.3 ms` `p50` (`407.2 ms` `p99`)** on warm uncontended calls (`4.8x` faster `p50`) and **`278.2 ms` `p50`** across the combined Tier 0+1 suite (`4.0x` faster `p50`, **`9.1x` lower router API cost**).
2. **`2.4x` More Net Wall-Clock Time Saved Than `GenerateContent` Routing:** On a 100-prompt workload, a traditional `GenerateContent` JSON router adds **`115.2 seconds`** of router latency overhead, consuming **69% of its downstream generation time savings** and leaving only **`12.1%`** net wall-clock speedup. Gemini Decisions Router adds only **`29.5 seconds`** across 100 prompts (with Tier 0 fast rules handling 11% of turns in `<0.1 ms`), delivering a **`29.1%` net wall-clock reduction** (`473.2s -> 335.5s`, saving **`137.7 seconds`** per 100 prompts) and **`56.3%` net API cost savings** even with downstream completions conservatively capped at `maxOutputTokens: 512`.
3. **Zero Frontier Under-Routing (`0.0%` Pro-Miss Rate):** Across all **30** ground-truth `pro` tasks (20 distributed architecture and concurrency debugging prompts, 5 complex causal/queueing data investigations, 3 Turn 2+ asymmetric thread-floor continuations, and 2 explicit overrides), Gemini Decisions Router achieved **`100.0%` Pro recall (`0/30` Pro-misses)** while routing **`0/70`** non-Pro prompts to `pro`. By comparison, the offline keyword heuristic missed **`36.7%` (`11/30`)** of `pro` tasks.

---

## Methodology & Dataset Design

### 1. Stratified 100-Prompt Dataset (`benchmarks/prompts_100.json`)

The benchmark evaluates 100 diverse, non-trivial engineering and agentic prompts across 6 categories:

- **`quick_lookup` (`20` prompts, expected `flash_lite`):** CLI flags (`tar`, `jq`, `git`, `kubectl`, `curl`), HTTP status codes, Python stdlib signatures, SQL syntax checks, port lookups, and regex snippets.
- **`coding_edit` (`20` prompts, expected `flash`):** Single-service engineering implementations including FastAPI validation, `pytest` fixtures, TypeScript state refactors, Go retry middleware, SQLite migrations, streaming CSV parsers, and multi-stage Dockerfiles.
- **`writing_docs` (`15` prompts, expected `flash`):** OpenAPI specs, incident postmortems, onboarding runbooks, ADR templates, release changelogs, PR descriptions, and migration guides.
- **`data_analysis` (`15` prompts, `10` expected `flash` + `5` expected `pro`):** 10 standard SQL cohort/retention/latency/billing queries (`flash`) plus 5 multi-system statistical and queueing root-cause analyses such as Simpson's paradox causal adjustment and GC/connection-pool tail-latency decomposition (`pro`).
- **`deep_debug_architecture` (`20` prompts, expected `pro`):** Multi-region database failover, Raft/Paxos split-brain recovery, lock-free ring buffer ABA memory-ordering bugs, zero-downtime envelope key rotation, OAuth2 token-binding threat models, and distributed saga idempotency under network partitions.
- **`agentic_lifecycle` (`10` prompts, `5` expected `flash_lite` + `5` expected `pro`):** Short operational confirmations (`"ship it"`, `"lgtm"`), Turn 2+ asymmetric thread-floor continuations (`"yes go ahead"` on a `pro`-locked thread), lightweight `PreToolUse` subagent briefs, and explicit model overrides (`"use gemini pro..."`).

Overall ground-truth tier distribution across the 100 prompts: **`25` `flash_lite` (25%)**, **`45` `flash` (45%)**, and **`30` `pro` (30%)**. Static `Always-Pro` therefore over-routes **`70%` of total traffic** (`100%` of the 70 non-Pro prompts).

### 2. Execution Protocol & Fairness Controls

- **Uncontended Sequential Timing (`--concurrency 1`):** All live calls ran sequentially (`concurrency=1`) after a 1-call TLS/auth warmup probe so `p50`, `p90`, `p95`, and `p99` percentiles reflect true single-request client wall-clock latency with zero local thread-pool contention (`0` HTTP retries occurred across the entire 169.8s live run).
- **Stratified End-to-End Sample (`N=15`, `sample_e2e: true`):** 5 `quick_lookup` (`flash_lite`), 5 `coding_edit`/`writing_docs`/`data_analysis` (`flash`), and 5 `deep_debug_architecture` (`pro`) prompts were executed end-to-end through real `generateContent` endpoints (`gemini-3.1-flash-lite-preview`, `gemini-2.5-flash`, and `gemini-2.5-pro`) with `maxOutputTokens: 512` across all tiers. Capping `maxOutputTokens` at `512` is conservative because unconstrained `pro` responses typically emit 1,500 to 4,000 tokens.
- **Standard Public Per-1M Token Pricing (USD):**
  - `gemini-decision-1`: `$0.10` input / `$0.00` output (`0` output tokens)
  - `flash_lite`: `$0.10` input / `$0.40` output
  - `flash` (and `llm_json_router` on `gemini-2.5-flash`): `$0.30` input / `$2.50` output (including billed `thoughtsTokenCount`)
  - `pro`: `$1.25` input / `$10.00` output (including billed `thoughtsTokenCount`)

---

## Reproducibility Commands

```bash
# 1. Run deterministic offline mock benchmark (<0.1s, zero network calls, CI-safe)
python3 benchmarks/run_benchmark.py --mock --output /tmp/mock_bench.json

# 2. Run full live 100-prompt + 15-prompt E2E benchmark (uses ~/.config/gemini-decisions-router/.env or ADC)
python3 benchmarks/run_benchmark.py \
  --dataset benchmarks/prompts_100.json \
  --output benchmarks/results_100.json \
  --concurrency 1

# 3. Render the 1600x1040 4-panel dark-theme dashboard PNG
python3 benchmarks/render_charts.py \
  --input benchmarks/results_100.json \
  --output assets/benchmark-dashboard.png
```

---

## Detailed Benchmark Results

### 1. Router Latency & Token Overhead

| Router Strategy | Calls (`N`) | Tier 0 Hits (`<1 ms`) | `min` (ms) | `p50` (ms) | `p90` (ms) | `p95` (ms) | `p99` (ms) | `max` (ms) | `mean` (ms) | Mean In / Out Tokens | Cost / 100 Prompts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`decisions_router_full` (Tier 0+1)** | `100` | `11` | `0.01` | **`278.2`** | **`411.4`** | `520.3` | `1,146.2` | `1,717.0` | `295.0` | `459.1` / **`0.0`** | **`$0.0046`** |
| **`decisions_tier1_only` (Pure `gemini-decision-1`)** | `100` | `0` | `131.7` | **`235.3`** | **`374.2`** | **`401.2`** | **`407.2`** | **`413.4`** | **`252.5`** | `513.1` / **`0.0`** | **`$0.0051`** |
| **`llm_json_router` (`gemini-2.5-flash` JSON)** | `15` | `0` | `511.4` | `1,124.3` | `1,643.8` | `1,901.5` | `2,112.3` | `2,165.0` | `1,151.6` | `186.9` / `144.3` | `$0.0417` |
| **`lexical_only` (Offline Fallback)** | `100` | `100` | `0.00` | `0.01` | `0.01` | `0.01` | `0.01` | `0.02` | `0.01` | `0.0` / `0.0` | `$0.0000` |

> [!NOTE]
> On the identical `N=15` single-turn sample (`sample_15_apples_to_apples`), pure `gemini-decision-1` achieved **`216.2 ms` `p50`** and Tier 0+1 achieved **`303.7 ms` `p50`** versus **`1,124.3 ms` `p50`** for `GenerateContent` JSON routing (`3.7x` to `5.2x` faster `p50`). Note also why `GenerateContent` JSON routing averaged `1,151.6 ms`: even with `temperature: 0.0` and `responseMimeType: "application/json"`, `gemini-2.5-flash` expended an average of `128.9` internal reasoning tokens before emitting `15.3` JSON tokens (`144.3` total billed output tokens per routing decision).

---

### 2. Per-Category Accuracy & Pro-Miss Analysis (`N=100`)

| Category | Prompts | Expected Tier(s) | Tier 0+1 Tier Acc | Pure Tier 1 Tier Acc | Lexical Tier Acc | Tier 0+1 `task_class` Acc | Tier 0+1 Pro-Misses | Lexical Pro-Misses |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`quick_lookup`** | `20` | `20` `flash_lite` | **`80.0%`** (`16/20`) | **`80.0%`** (`16/20`) | `65.0%` (`13/20`) | `95.0%` | `0 / 0` | `0 / 0` |
| **`coding_edit`** | `20` | `20` `flash` | **`100.0%`** (`20/20`) | **`100.0%`** (`20/20`) | `100.0%` (`20/20`) | `100.0%` | `0 / 0` | `0 / 0` |
| **`writing_docs`** | `15` | `15` `flash` | **`100.0%`** (`15/15`) | **`100.0%`** (`15/15`) | `86.7%` (`13/15`) | `100.0%` | `0 / 0` | `0 / 0` |
| **`data_analysis`** | `15` | `10` `flash`, `5` `pro` | **`100.0%`** (`15/15`) | **`100.0%`** (`15/15`) | `93.3%` (`14/15`) | `66.7%` | **`0 / 5`** | `1 / 5` |
| **`deep_debug_architecture`** | `20` | `20` `pro` | **`100.0%`** (`20/20`) | **`100.0%`** (`20/20`) | `60.0%` (`12/20`) | `100.0%` | **`0 / 20`** | **`8 / 20`** |
| **`agentic_lifecycle`** | `10` | `5` `flash_lite`, `5` `pro` | **`100.0%`** (`10/10`) | **`100.0%`** (`10/10`) | `70.0%` (`7/10`) | `100.0%` | **`0 / 5`** | `2 / 5` |
| **Total / Overall** | **`100`** | `25` / `45` / `30` | **`96.0%`** (`96/100`) | **`96.0%`** (`96/100`) | **`79.0%`** (`79/100`) | **`94.0%`** (`93.3%` 5-way) | **`0 / 30` (`0.0%`)** | **`11 / 30` (`36.7%`)** |

#### 3x3 Tier Confusion Matrix (`decisions_router_full`, `N=100`)

| Ground Truth \ Routed Tier | Routed `flash_lite` | Routed `flash` | Routed `pro` | Row Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Expected `flash_lite` (`25`)** | **`21`** | `4` | `0` | `84.0%` (`0` under-routed, `0` sent to `pro`) |
| **Expected `flash` (`45`)** | `0` | **`45`** | `0` | **`100.0%`** |
| **Expected `pro` (`30`)** | `0` | `0` | **`30`** | **`100.0%`** (`0.0%` Pro-Miss Rate) |

**Error & Boundary Analysis:**
- **Zero Under-Routing (`0/100`):** Neither `decisions_router_full` nor `decisions_tier1_only` ever routed a `pro` prompt to `flash`/`flash_lite` or a `flash` prompt to `flash_lite`.
- **Conservative Borderline Escalation (`4/100`):** All 4 routing differences across the 100-prompt suite occurred on multi-clause `quick_lookup` questions (regex lookarounds or multi-flag CLI commands) where `effort_rubric` scored slightly above `0.85`, nudging the route one step up from `flash_lite` to `flash` rather than risking a truncated answer.
- **Multi-Signal Fusion (`data_analysis`):** On the 5 complex causal/queueing `data_analysis` prompts, `gemini-decision-1` classified `task_class` as `deep_debug_architecture` or set `P(needs_pro_reasoning) >= 0.55` and `effort_rubric >= 2.25`, achieving **`100.0%` (`15/15`)** exact tier separation between the 10 `flash` SQL tasks and the 5 `pro` root-cause investigations.

---

### 3. Confidence Calibration (`decisions_tier1_only`, `N=100`)

`gemini-decision-1` returns native per-question confidence scores without sampling multiple completions. Combining confidence across `task_class` (`0.45`), `needs_pro_reasoning` (`0.35`), and `effort_rubric` (`0.20`) yields well-ordered uncertainty estimates (`ECE = 0.1179` on pure Tier 1, **`ECE = 0.1031`** on Tier 0+1):

| Confidence Bin | Prompt Count (`Pure Tier 1`) | Mean Predicted Confidence | Empirical Tier Accuracy | Calibration Gap (`|Acc - Conf|`) |
| :--- | :---: | :---: | :---: | :---: |
| **`0.00 - 0.70`** | `9` | `64.1%` | `88.9%` | `0.2476` *(under-confident, safe)* |
| **`0.70 - 0.80`** | `17` | `75.9%` | `94.1%` | `0.1820` *(under-confident, safe)* |
| **`0.80 - 0.90`** | `48` | `85.6%` | `97.9%` | `0.1232` |
| **`0.90 - 1.00`** | `26` | `94.0%` | **`96.2%`** | **`0.0212`** |

Notice the directional safety property: across every bin, empirical accuracy **exceeds** predicted confidence (mildly conservative under-confidence rather than dangerous over-confidence), and the top confidence bin (`>= 0.90`) exhibits a calibration gap of just **`2.1%`**. This makes `min_confidence` thresholds (`0.45` default, `0.55` recommended for strict `auto_switch`) reliable in production.

---

### 4. End-to-End Wall-Clock Time & API Cost Savings

#### Downstream Tier Completion Baselines (`maxOutputTokens: 512`)

| Target Tier | Probed Live Model | Mean Completion Latency | `p50` Completion Latency | Mean Prompt / Output Tokens | Mean Completion Cost |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`flash_lite`** | `gemini-3.1-flash-lite-preview` | **`1,435.0 ms`** (`3.3x` faster) | `1,399.2 ms` | `20.2` / `352.2` | **`$0.000143`** (`36x` cheaper) |
| **`flash`** | `gemini-2.5-flash` | **`2,726.9 ms`** (`1.7x` faster) | `2,747.4 ms` | `48.7` / `508.0` | **`$0.001285`** (`4.0x` cheaper) |
| **`pro`** | `gemini-2.5-pro` | `4,742.6 ms` (`1.0x`) | `4,673.2 ms` | `48.9` / `509.0` | `$0.005151` (`1.0x`) |

#### Empirical 15-Prompt Sample & Projected 100-Prompt Workload

| Workload & Strategy | Router Overhead (s) | Generation Time (s) | Total E2E Wall Time (s) | Net Time Saved vs Always-Pro | Router Cost (USD) | Generation Cost (USD) | Total E2E Cost (USD) | Net Cost Saved vs Always-Pro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15-Prompt Live Sample: Always-Pro** | `0.00 s` | `71.14 s` | `71.14 s` | `0.0%` | `$0.0000` | `$0.0773` | `$0.0773` | `0.0%` |
| **15-Prompt Live Sample: `GenerateContent`-Routed** | `17.27 s` | `46.22 s` | `63.50 s` | `-10.7%` | `$0.0063` | `$0.0330` | `$0.0393` | `-49.1%` |
| **15-Prompt Live Sample: Gemini Decisions Router** | **`4.57 s`** | `46.95 s` | **`51.52 s`** | **`-27.6%`** | **`$0.0008`** | `$0.0341` | **`$0.0349`** | **`-54.8%`** |
| **100-Prompt Workload: Always-Pro** | `0.00 s` | `473.21 s` | `473.21 s` | `0.0%` | `$0.0000` | `$0.5154` | `$0.5154` | `0.0%` |
| **100-Prompt Workload: `GenerateContent`-Routed** | `115.16 s` | `300.86 s` | `416.02 s` | `-12.1%` | `$0.0417` | `$0.2159` | `$0.2576` | `-50.0%` |
| **100-Prompt Workload: Gemini Decisions Router** | **`29.50 s`** | `306.03 s` | **`335.53 s`** | **`-29.1%`** | **`$0.0046`** | `$0.2205` | **`$0.2251`** | **`-56.3%`** |
