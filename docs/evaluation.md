# Evaluation

`python -m scripts.evaluate` runs offline over 5 SME business cases (`data/eval/cases.jsonl`).

**Chunking fix, before and after** (same corpus, same embedder):

| Metric | Before | After |
|---|---|---|
| Chunks | 52 | 205 |
| Corpus tokens inside the embedder's 256-token window | 36% | **99%** |
| Field-query precision@3 (7 Canvas fields; share of top-3 passages that discuss the field) | 0.19 | **0.48** |
| Median retrieval latency (CPU) | 17 ms | 20 ms |

pypdf output rarely contains blank lines, so the chunker treated each page as one paragraph. Chunks reached several thousand characters, and MiniLM silently truncated them, so most of each page was invisible to retrieval.

**Per-case retrieval** (top-6, query = business case text):

| Case | Framework papers in top-6 (of 3) | Share of passages from the example roadmap |
|---|---|---|
| alexandra_bikeshop | 0 | 1.00 |
| family_bakery | 2 | 0.50 |
| regional_logistics | 2 | 0.33 |
| dental_clinic_group | 2 | 0.50 |
| industrial_parts_distributor | 1 | 0.83 |

The worked bikeshop roadmap (`corpus/bikeshop_roadmap/`) is indexed alongside the framework papers and dominates retrieval, especially for the bikeshop case itself. See the limitations in the [README](../README.md#limitations).

**Live pipeline metrics** (`python -m scripts.evaluate --live`, needs `GOOGLE_API_KEY`). Run on 2026-10-10 with `gemini-2.5-flash-lite` (fast tier) and `gemini-2.5-flash` (powerful tier), free tier:

| Case | Agents with valid JSON | Canvas priority fields in expected set | Evaluator score (1–10) | Latency (s) |
|---|---|---|---|---|
| alexandra_bikeshop | 6/6 | 0.67 | 6 | 47.5 |
| family_bakery | 6/6 | 0.67 | 6 | 45.6 |
| regional_logistics | 6/6 | 0.67 | 6 | 42.7 |
| dental_clinic_group | not run: daily quota | | | |
| industrial_parts_distributor | not run: daily quota | | | |
| **3 cases** | **18/18** | **0.67** | **6.0** | **45.3 mean** |

The free-tier daily cap on `gemini-2.5-flash` ran out during the fourth case, after 16 successful requests that day. The router fell back to the mock as designed, so those two rows measure nothing and are left out. Latency is the sum of stage timings; in the recorded bikeshop run the four powerful-tier stages (canvas, strategist, roadmap, evaluator) took 87% of it.

Before the thinking-budget fix, this table would have read 2/6: `gemini-2.5-flash` thinks by default, and thinking tokens count against `max_output_tokens`. With the old 2048 limit it spent ~1960 tokens thinking and cut every powerful-tier answer off after ~80 tokens. The powerful tier now gets `GEMINI_THINKING_BUDGET` (1024) on top of a 4096-token answer budget, and the fast tier runs with thinking off.
