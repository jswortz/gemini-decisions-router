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
# 1. Run deterministic offline mock benchmark (<0.1s, zero network calls, streams live stdout stats)
python3 benchmarks/run_benchmark.py --mock --output /tmp/mock_bench.json

# 2. Run deterministic offline N=100 flash-lite & flash JSON head-to-head (--quiet for summary box only)
python3 benchmarks/run_router_head_to_head.py --mock --output /tmp/mock_h2h.json
python3 benchmarks/run_router_head_to_head.py --mock --quiet --output /tmp/mock_h2h.json

# 3. Run full live 100-prompt + 15-prompt E2E benchmark (uses ~/.config/gemini-decisions-router/.env or ADC)
python3 benchmarks/run_benchmark.py \
  --dataset benchmarks/prompts_100.json \
  --output benchmarks/results_100.json \
  --concurrency 1

# 4. Run full live N=100 flash-lite & flash JSON router head-to-head benchmark
python3 benchmarks/run_router_head_to_head.py \
  --dataset benchmarks/prompts_100.json \
  --output benchmarks/results_flash_lite_router_100.json \
  --concurrency 1

# 5. Render the 1600x1040 4-panel dark-theme dashboard PNG
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

---

### 5. `N=100` Head-to-Head Study: `gemini-decision-1` vs `flash-lite` & `flash` JSON Routers

A common engineering question is: *"What if we disable thinking (`thinkingBudget: 0`), enforce `responseMimeType: "application/json"`, and use `gemini-3.1-flash-lite-preview` or `gemini-2.5-flash` as a lightweight structured-output router across all 100 prompts?"*

We executed that exact 100-prompt head-to-head live in `benchmarks/run_router_head_to_head.py` (`benchmarks/results_flash_lite_router_100.json`, sequential `concurrency=1`, `0` HTTP retries):

| Head-to-Head Metric (`N=100`) | Gemini Decisions Router (Tier 0+1) | Pure `gemini-decision-1` (Tier 1) | `flash-lite` JSON Router (`gemini-3.1-flash-lite-preview`, `thinkingBudget=0`) | `flash` JSON Router (`gemini-2.5-flash`, `thinkingBudget=0`) | `flash` JSON Router (`gemini-2.5-flash`, default thinking, `N=15`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Router Latency `p50`** | **`278.2 ms`** | **`235.3 ms`** | `709.4 ms` (**`2.55x-3.01x` slower**) | `480.0 ms` (`1.73x-2.04x` slower) | `1,124.3 ms` (`4.04x-4.78x` slower) |
| **Router Latency `p90` / `p99`** | `411.4 ms` / `1,146.2 ms` | **`374.2 ms` / `407.2 ms`** | `868.9 ms` / `1,014.2 ms` | `552.1 ms` / `606.2 ms` | `1,643.8 ms` / `2,112.3 ms` |
| **Mean In / Out Tokens** | `459.1` / **`0.0`** | `513.1` / **`0.0`** | `228.1` / `20.7` (`0` thought) | `228.1` / `24.1` (`0` thought) | `186.9` / `144.3` (`128.9` thought) |
| **Exact 3-Tier Accuracy** | **`96.0%`** (`96/100`) | **`96.0%`** (`96/100`) | `93.0%` (`93/100`) | `93.0%` (`93/100`) | `100.0%` (`15/15` single-turn) |
| **Critical Pro-Miss Rate** | **`0.0%`** (`0/30`) | **`0.0%`** (`0/30`) | **`10.0%`** (`3/30` missed) | **`10.0%`** (`3/30` missed) | `0.0%` (`0/5` single-turn) |
| **Non-Pro Over-Routed to Pro** | **`0.0%`** (`0/70`) | **`0.0%`** (`0/70`) | `5.7%` (`4/70`) | `5.7%` (`4/70`) | `0.0%` (`0/10`) |
| **Confidence Bins Populated (`4` bins)** | **`4 / 4`** (`0.66 -> 0.94`) | **`4 / 4`** (`0.64 -> 0.94`) | **`1 / 4`** (`100/100` in `[0.90, 1.00]`) | **`1 / 4`** (`100/100` in `[0.90, 1.00]`) | `0 / 4` (no probabilities) |
| **Mean Confidence When Wrong** | `0.814` (`Brier = 0.0524`) | `0.814` (`Brier = 0.0583`) | **`0.943`** (`7/7` errors `>= 0.85`, `Brier = 0.0634`) | **`0.900`** (`7/7` errors `>= 0.85`, `Brier = 0.0633`) | N/A |
| **Router Cost / 100 Prompts** | **`$0.00459`** | **`$0.00513`** | `$0.00881` (`$0.25/$1.50` preview rates; `$0.00311` at `$0.10/$0.40`) | `$0.01288` (`2.80x` higher) | `$0.04168` (`9.08x` higher) |
| **Projected 100-Prompt E2E Time** | **`335.53 s` (`-29.1%`)** | **`331.27 s` (`-30.0%`)** | `372.84 s` (`-21.2%`, `+37.3 s` slower) | `346.81 s` (`-26.7%`) | `416.02 s` (`-12.1%`) |

**Why Single-Pass Prefill Outperforms `flash-lite` Structured JSON Routing:**
1. **`2.55x` to `3.01x` Faster `p50` Latency:** Even with `thinkingBudget: 0` and `temperature: 0.0`, `gemini-3.1-flash-lite-preview` still had to autoregressively decode `20.7` JSON tokens per call (`709.4 ms` `p50`, `1,014.2 ms` `p99`), adding **`71.3 seconds`** of router overhead per 100 prompts vs **`25.3s` to `29.5s`** for `gemini-decision-1`.
2. **Zero Frontier Drops vs `10.0%` Stateless Pro-Misses:** Both `flash-lite` and `flash` JSON routers dropped all 3 Turn 2+ `pro`-locked continuation prompts (`"yes go ahead"`, `"do step 2 next"`, `"sounds good, proceed"`) down to `flash_lite`/`flash` (`10.0%` Pro-Miss Rate) and over-routed `4/70` `flash` tasks up to `pro` (`5.7%` over-route rate).
3. **Verbalized Overconfidence Defeats Confidence Gating:** Asking `flash-lite` or `flash` to emit `"confidence"` inside JSON collapsed **100 out of 100 prompts into the `[0.90, 1.00]` bin** (`stdev = 0.023`), assigning **`0.943` mean confidence when wrong** (`100%` of errors had `confidence >= 0.85`). By contrast, `gemini-decision-1` derives confidence from single-pass prefill posteriors across `4/4` populated bins (`Brier = 0.0524`), enabling reliable `min_confidence` gating.

---

## 6. Comparison Against Leading LLM Routing Frameworks

| Engineering Dimension | Gemini Decisions Router (`gemini-decision-1`, Tier 0+1) | RouteLLM (LMSYS `mf`, `sw_ranking`, `bert`, `causal_llm`) | Semantic Router (Aurelio AI) | Arch-Router-1.5B (ArchGW) & vLLM Semantic Router | LangGraph / LlamaIndex / PydanticAI (`flash-lite` / `flash` JSON) | Commercial Hosted Routers (NotDiamond, Martian, `openrouter/auto`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Mechanism** | **Tier 0** local rules (`<0.1 ms`) + **Tier 1** 1-pass prefill over 3 rubric questions (`choice` + `noul` + `score`) | Pairwise `P(strong_wins)` via Matrix Factorization, Elo, `BERT`, or `Llama-3-8B` ([arXiv:2406.18665](https://arxiv.org/abs/2406.18665)) | Dense/sparse encoder + cosine similarity over static example `utterances` | **Arch-Router:** `1.5B` JSON decode ([arXiv:2506.16655](https://arxiv.org/abs/2506.16655)). **vLLM SR:** `ModernBERT` Envoy filter ([arXiv:2510.08731](https://arxiv.org/abs/2510.08731)) | Autoregressive `GenerateContent` constrained to Pydantic / JSON schema (`responseMimeType: "application/json"`) | **NotDiamond:** SaaS meta-classifier. **Martian:** Model Mapping proxy. **OpenRouter:** ~30-task market router |
| **Router Output Tokens** | **`0` output tokens** (`0` thought tokens) | `0` (`mf`, `sw_ranking`, `bert`); `1` token (`causal_llm`) | **`0` output tokens** (vector embedding) | **Arch-Router:** `~8-15` JSON tokens. **vLLM SR:** `0` tokens | **`20.7` (`flash-lite`, `thinking=0`) to `144.3` (`flash` default)** | `0` client-visible tokens (server-side proxy) |
| **Routing Latency (`p50` / `p99`)** | **`235-278 ms` `p50` / `407-1,146 ms` `p99`** (Tier 0 `<0.1 ms`) | `~6-24 ms` on dedicated L4 GPU + `~50-150 ms` OpenAI embed RTT (`~345 ms` CPU on `sw_ranking`) | `~5-15 ms` (local ONNX) or `~45-120 ms` (cloud embedding API) | **Arch-Router:** `~51 ms` GPU (`~250-600 ms` CPU). **vLLM SR:** `~5-20 ms` local Envoy filter | **`480-709 ms` `p50` / `606-1,014 ms` `p99`** (`thinking=0`); **`1,124 ms` `p50` / `2,112 ms` `p99`** (default thinking) | **NotDiamond:** `~100-150 ms` RTT. **Martian / OpenRouter:** `~20-80 ms` in-gateway |
| **Zero-Shot Rubric Flexibility** | **High:** `1-64` arbitrary `noul`, `choice` (`<=255` options), and `score` questions in 1 call; zero retraining | **Low:** Binary strong/weak split trained on ~80k Arena pairs; new tiers/models require re-fitting | **Low-Medium:** Requires curating `10-50+` utterances per route; conflates topic similarity with reasoning depth | **Arch-Router:** Medium-High (domain/action JSON policies). **vLLM SR:** Low (requires `ModernBERT` fine-tuning) | **High:** Arbitrary JSON schema in prompt, but latency scales linearly with output fields | **Low:** Fixed `cost`/`quality`/`latency` sliders; custom routers require labeled eval sets |
| **Calibration & Confidence Gating** | **Calibrated Posteriors:** `4/4` bins populated, `ECE = 0.103`, `Brier = 0.0524`, conservative under-confidence | Scalar pairwise win score (`P(strong_wins)`); uncalibrated across multi-rubric tiers | Uncalibrated cosine similarity (`[-1, 1]`); requires `RouteLayer.fit()` threshold sweeps | **Arch-Router:** Uncalibrated greedy JSON text. **vLLM SR:** Raw `ModernBERT` softmax logits | **Verbalized Overconfidence:** `100/100` collapsed in `[0.90, 1.00]`, `0.90-0.94` mean conf when wrong ([arXiv:2306.13063](https://arxiv.org/abs/2306.13063)) | **Opaque:** Proxy returns winning model ID without pre-dispatch rubric posteriors |
| **Agentic Lifecycle Awareness** | **Native:** Smart Turn 2+ thread floor, `PreToolUse` `invoke_subagent` right-sizing, `agy models` + LS auto-discovery | **Stateless:** Downgrades Turn 2+ `"yes go ahead"` replies, busting KV caches and losing Pro context | **Stateless:** No conversation floor, subagent tool-hook interception, or model discovery | **Gateway-Only:** Inspects HTTP payloads in Envoy, but lacks IDE `PreToolUse` hooks or CLI discovery | **DIY:** Requires hand-coding thread state, subagent hooks, and JSON fallback guards in app code | **Session Stickiness Only:** Basic provider failover, no client hook or subagent awareness |
| **Infra Footprint & Cost / `1,000` Routes** | **Pure Python 3.11+ stdlib (`0` pip pkgs, `0` GPUs)**; **`~$0.046` / 1k routes** | PyTorch + `3-16 GB` GPU (`~$500-$800/mo` L4 idle cost at dev QPS) + embedding API | `semantic-router` SDK + `100-500 MB` ONNX encoder (`$0` local or `~$0.01/1k` API) | Envoy + Docker/K8s + dedicated GPU/Rust daemon (`$150-$800/mo` cluster footprint) | Framework SDK + **`~$0.03-$0.09` / 1k** (`flash-lite`) up to **`~$0.42` / 1k** (`flash`) | Third-party SDK (`~$0.025-$1.00 / 1k` on NotDiamond) or external multi-tenant proxy egress |
