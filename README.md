*This project has been created as part of the 42 curriculum by ydimitra.*

# RAG Against the Machine — `semantic-hybrid` branch

![42](https://img.shields.io/badge/42-Codam-000000?style=flat)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?style=flat)
![bm25s](https://img.shields.io/badge/lexical-bm25s-blue?style=flat)
![sentence-transformers](https://img.shields.io/badge/semantic-MiniLM--L6--v2-green?style=flat)
![bonus](https://img.shields.io/badge/bonus-%231%20%232%20%234-purple?style=flat)

*Fusing BM25 with a bi-encoder, rank by rank — plus the two caches that make repeated queries on this branch nearly free.*

This is the **bonus branch**. It carries the full mandatory system plus semantic
embeddings (#1), hybrid RRF fusion (#2), and a two-level cache (#4), and is kept
separate from `main` on purpose — see [What is this branch?](#-what-is-this-branch)
for why.

## 📑 Table of Contents
- [What is this branch?](#-what-is-this-branch)
- [Project Structure — what's new here](#-project-structure--whats-new-here)
- [Reading Order — Where to Start](#-reading-order--where-to-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [The Makefile](#-the-makefile)
- [Algorithm Explanation — Semantic Embeddings, Hybrid RRF & Caching](#-algorithm-explanation--semantic-embeddings-hybrid-rrf--caching)
- [Design Decisions](#-design-decisions)
- [Features Implemented](#-features-implemented)
- [Performance Analysis](#-performance-analysis)
- [Challenges Faced](#-challenges-faced)
- [Testing Strategy](#-testing-strategy)
- [Example Usage](#-example-usage)
- [Resources](#-resources)

## 🧩 What is this branch?

`main` is the validated mandatory system: lexical (BM25) retrieval only, measured at
docs recall@5 0.8200 / code recall@5 0.7576, both clear of the subject's bars. This
branch adds three bonuses on top of it — a semantic (embedding) retriever, a hybrid
mode that fuses semantic with lexical via Reciprocal Rank Fusion, and a two-level cache
covering all three modes — without touching `main`'s mandatory behaviour.

It's kept as a separate branch rather than merged, for a specific, measured reason:
**hybrid underperforms pure lexical on both public datasets** (0.7800/0.7273 vs.
0.8200/0.7576 — see [Performance Analysis](#-performance-analysis)). `main` must stay
the graded, tuned baseline; this branch is where the bonus work lives, demonstrable and
independently measured, without putting the mandatory bar at risk.

This README is deliberately a **delta**, not a duplicate — it documents what's added or
changed on this branch only. For the mandatory-only description, chunking/tokenizer
detail, and BM25 design rationale, see **`main`**'s README; nothing there is repeated
here.

## 🗂 Project Structure — what's new here

Same top-level layout as `main` (`src/`, `tests/`, `docs/`, `data/`), plus:

<details>
<summary><strong>📁 src/</strong> — two new modules on top of <code>main</code></summary>

| file | purpose |
|---|---|
| `embeddings.py` | *new* — loads `all-MiniLM-L6-v2`, embeds/persists/queries the `(n_chunks, 384)` matrix, semantic top-k via cosine similarity |
| `cache.py` | *new* — sqlite query-results cache keyed on `(query, k, mode)`, fingerprinted on the index files' mtime+size |
| `retriever.py` | *changed* — gained `mode: "lexical" \| "semantic" \| "hybrid"` and `fuse()` (Reciprocal Rank Fusion) |
| `indexer.py` | *changed* — `bm25s.BM25.load(..., mmap=True)` fast-load path for the index cache |
| `__main__.py` | *changed* — every command gained an optional `--mode` flag; cache is checked before the embedding model loads |

Every other `src/` file (`models.py`, `tokenizer.py`, `chunking.py`, `generator.py`,
`evaluator.py`) is unchanged from `main`.

</details>

<details>
<summary><strong>📁 data/processed/</strong> — two new artifacts (gitignored, local only)</summary>

```
data/processed/
├── index.pkl / bm25s postings   # unchanged from main
├── embeddings.npy               # new — (28246, 384) float32 matrix, L2-normalized
└── query_cache.db               # new — sqlite, (query,k,mode) -> ranked results
```

</details>

## 📖 Reading Order — Where to Start

Assumes you've already read `main`'s modules (`models.py` → `tokenizer.py` →
`chunking.py` → `indexer.py` → `retriever.py` → `generator.py` → `evaluator.py`). On
top of that, for this branch:

1. **`embeddings.py`** — how a chunk becomes a 384-dim row, and how a query becomes a
   ranked list via a single matrix-vector product.
2. **`retriever.fuse()`** — Reciprocal Rank Fusion: how two independently-ranked lists
   (lexical, semantic) become one.
3. **`cache.py`** — the `(query, k, mode)` key, the fingerprint invalidation rule, and
   why a miss must never raise.
4. **`__main__.py`** — where the cache check is wired in *before* `_load_semantic()`,
   the detail that makes a warm hit skip loading the embedding model entirely.

## ⚙️ Installation

Same as `main`: Python 3.10+, [`uv`](https://docs.astral.sh/uv/).

```bash
make install && make lint && make test
```

Place the corpus under `data/raw/vllm-0.10.1/` and datasets under
`data/datasets/{UnansweredQuestions,AnsweredQuestions}/` before indexing (`data/` stays
gitignored). One thing specific to this branch: `--mode semantic`/`hybrid` on `index`
also builds `data/processed/embeddings.npy` — budget ~5 minutes combined, see
[Performance Analysis](#-performance-analysis).

## ▶️ Usage

Same six commands as `main`, all gaining an optional `--mode lexical|semantic|hybrid`
(default `lexical`, reproduces the mandatory behaviour exactly — byte-identical output
to `main` when omitted):

| command | new flag on this branch |
|---|---|
| `index --max_chunk_size <int> --mode ...` | `semantic`/`hybrid` also builds `embeddings.npy` |
| `search <query> --k <int> --mode ...` | checks the query cache before loading any model |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir> --mode ...` | same cache check, per question |
| `answer <query> --k <int> --mode ...` | |
| `answer_dataset` / `evaluate` | unchanged from `main` |

## 🛠 The Makefile

Same targets as `main`, with one difference: `lint`/`lint-strict` cover `tests/` too on
this branch (`main` lints `src` only).

| command | what it does |
|---|---|
| `make install` | `uv sync` |
| `make run` | Runs `uv run python -m src` |
| `make debug` | Runs the CLI under `pdb` |
| `make lint` | `flake8` + `mypy` over `src tests` |
| `make lint-strict` | `mypy --strict --ignore-missing-imports` over `src tests` |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make test` | Runs the pytest suite |

## 🧠 Algorithm Explanation — Semantic Embeddings, Hybrid RRF & Caching

**Semantic retrieval (`embeddings.py`):** every chunk is embedded with
`all-MiniLM-L6-v2` (384-dim, CPU-friendly), rows L2-normalized at encode time so cosine
similarity reduces to a single matrix-vector dot product at query time
(`embeddings @ query_vec`, `np.argpartition` for top-k rather than a full sort). Chunk
text is prefixed with its file path before embedding — the semantic-side equivalent of
`main`'s path-token indexing, since a bi-encoder has no other way to see a filename; this
alone lifted semantic recall@5 from docs 0.53→0.58, code 0.37→0.52.

**Hybrid fusion (`retriever.fuse()`):** Reciprocal Rank Fusion of the lexical and
semantic rankings — `RRF(d) = Σ 1/(c + rank_i(d))` summed over both retrievers,
1-based ranks, `c=60` (the Cormack et al. standard constant), top-100 candidates pulled
from each side before fusing, sorted desc with a `(score, chunk_id)` tie-break. Rank-based
fusion needs no cross-scale normalization between BM25 scores and cosine similarities —
simpler to implement and to defend than score-mixing.

**Caching (`cache.py` + `indexer.py`):** two independent caches, per subject ch. IX #4.

- **Index cache (cold start):** `bm25s.BM25.load(..., mmap=True)` memory-maps the
  postings instead of reading them eagerly — the format chosen back in Phase 4 (pickle
  metadata + `bm25s`' own npy/json, not one big pickle) was already fast; mmap is free
  extra headroom.
- **Query-results cache (repeated queries):** a sqlite table maps
  `sha256(mode ∥ k ∥ query)` → the `(chunk_id, score)` list. `mode` and `k` are *inside*
  the key so a hybrid result can never be served for a lexical query, or `k=10` for
  `k=5` — one table for every mode, never fuzzy-matched. Each row is fingerprinted on the
  mtime+size of `index.pkl` (+ `embeddings.npy` when present); re-indexing changes the
  fingerprint, so stale rows stop matching and `put()` sweeps them.
- **The cache is checked before anything heavy loads.** The obvious place to cache is
  inside `top_k()`, but by then the embedding model is already in memory. Since loading
  `SentenceTransformer` costs ~4.3s against ~30ms for the index, the lookup happens in
  `__main__.py` *before* `_load_semantic()` runs — the reason a warm hit skips model
  loading entirely, not just re-scoring.
- **A missing or corrupt cache is a miss, never an error.** `get()` swallows database
  and OS errors and returns `None`; the CLI degrades to a normal cold query rather than
  crashing.

## 🧭 Design Decisions

Bonus-specific decisions, on top of `main`'s (see that README for the mandatory-side
log):

- **Chunk text re-sliced from disk at embed time, not stored** — cache text alongside
  the embedding matrix — same invariant as the lexical index: offsets are ground truth,
  and one extra read per file (not per chunk) is cheap next to the encode cost.
- **`np.argpartition` for semantic top-k, not a full sort** — mirrors `main`'s
  `heapq.nlargest` discipline: O(n) partial partition over ~28k rows, sort only the
  k-sized slice.
- **RRF over score-mixing** — normalizing BM25 and cosine onto a shared scale — rank-based
  fusion needs no cross-scale normalization and matches the subject's suggested formula;
  simpler to defend.
- **`mode` threaded through `retriever.top_k`/`search_dataset`, default `"lexical"`** —
  a separate hybrid-only function — keeps the mandatory path byte-identical to `main`
  (verified: a default-mode call takes the same branch as before this change) while
  every command gains an optional `--mode` flag.
- **sqlite3 for the query cache** — shelve/json — stdlib, no new dependency, a normal
  key-value table under `data/processed/`.
- **Cache check happens before the model loads, not inside `top_k`** — wrap caching
  around `top_k` internally — if `top_k` did its own cache-around, the embedding model
  would already be loaded by the time it's called, so a hit would still pay the 4.3s
  cost it exists to avoid.

## Features Implemented

This branch *is* the verification — everything below runs directly from here.

| Feature | Status | Where |
|---|---|---|
| Semantic embeddings (bonus #1) | ✅ built & measured | `src/embeddings.py` |
| Hybrid retrieval / RRF fusion (bonus #2) | ✅ built & measured | `retriever.fuse()` |
| Caching — index + query-results (bonus #4) | ✅ built & measured | `src/cache.py`, `indexer.py` mmap load |

### ✅ How to verify

```bash
uv run python -m src index --max_chunk_size 2000 --mode hybrid
uv run python -m src search "How to configure the OpenAI server?" --k 5 --mode hybrid
uv run python -m src search "How to configure the OpenAI server?" --k 5 --mode hybrid  # warm cache, ~24x faster
```

The second identical `search` call should return instantly and print byte-identical
results — that's the query-results cache hit, skipping the embedding model load
entirely (see timings below).

## 📊 Performance Analysis

Measured with the real CLI against both public datasets (k=5, IoU>0.05 match bar):

| mode | docs recall@5 | code recall@5 |
|---|---|---|
| lexical (same as `main`) | 0.8200 (82/100) | 0.7576 (75/99) |
| semantic | 0.5800 | 0.5152 |
| hybrid | 0.7800 | 0.7273 |

Chunk-size tuning (2000/600 vs. 1000/200) is mandatory-side work — see `main`'s README.

**Timings** (whole vLLM corpus, 28,246 chunks, real CLI runs):

| operation | time | budget |
|---|---|---|
| index (lexical only) | ~4.3s | ≤ 5 min |
| index (+ embeddings, `--mode semantic`/`hybrid`) | ~300s (296.8s embed step) | ≤ 5 min |
| single query, hybrid, cold cache | 9.27s | — |
| single query, hybrid, **warm cache** | **0.38s** (~24x) | — |
| `search_dataset`, 100 questions, hybrid, cold cache | 12.39s | — |
| `search_dataset`, 100 questions, hybrid, **warm cache** | **0.40s** (~31x) | — |

Warm-cache runs skip loading the embedding model entirely (100% cache hit) — output
verified byte-identical to the cold run via `diff`.

## 🧩 Challenges Faced

**Hybrid initially made recall *worse*, not better.** First implementation (raw chunk
text embedded, no path signal) scored hybrid at 0.72/0.6667 — *below* pure lexical on
both datasets, the opposite of the expected "hybrid lifts code recall" result. Two
hypotheses were tested independently, each measured before/after rather than assumed:

1. *Was the embedding model truncating our chunks?* `all-MiniLM-L6-v2` defaults to a
   256-token cap; our chunks run up to 2000 chars (~400-500+ tokens). Raising
   `max_seq_length` to 512 (the underlying transformer's architectural limit) ran
   without error — but measured **worse**: semantic recall dropped further and the embed
   step nearly doubled (278s → 528s, alone over the 5-minute budget). The model was
   fine-tuned specifically on ≤256-token sequences, so positions past that are
   out-of-distribution. Reverted.
2. *Was the semantic side missing the path-token signal that made lexical strong on
   code?* Prefixing each chunk's file path before embedding (keeping the 256-token
   default) measured a real gain: semantic recall docs 0.53→0.58, **code 0.37→0.52**.
   Hybrid followed: 0.72/0.6667 → **0.78/0.7273**.

Hybrid still trails pure lexical after the fix — a much weaker retriever fused in at
equal rank-weight (plain RRF has no weighting knob) drags a strong one down rather than
purely complementing it, especially once both retrievers share the same path-token
signal instead of it being lexical-exclusive. This is reported as a real, honest result
rather than hidden or force-tuned past it: hybrid is correctly implemented and
demonstrable, just not a recall win on this corpus with this embedding model — which is
exactly why it stays a branch, not a merge into `main`.

**Cache had to be wired around the model load, not around retrieval math.** The naive
place to cache is inside `top_k()` — but by the time `top_k()` is called, the embedding
model (the actual ~4.3s cold-start cost) has already been loaded. The cache check had to
move to the CLI/`search_dataset` boundary, *before* `_load_semantic()` runs, so a cache
hit can skip loading the model entirely rather than just skip re-scoring.

## 🧪 Testing Strategy

The mandatory suite (`test_chunking.py`, `test_tokenizer.py`, `test_indexer.py`,
`test_retriever.py`, `test_evaluator.py`, `test_models.py`, `test_cli.py`) carries over
unchanged from `main` and still passes on this branch (`make test`). **`embeddings.py`
and `cache.py` have no dedicated pytest coverage yet** — bonus correctness here was
validated by direct measurement instead (recall@5 per mode, cold-vs-warm byte-identical
`diff`, fingerprint-invalidation check on a bumped `index.pkl` mtime — see
[Design Decisions](#-design-decisions) and [Performance Analysis](#-performance-analysis)).
Worth adding before a live demo if time allows.

## 💡 Example Usage

Mandatory commands are identical to `main` (see that README). Bonus:

```bash
uv run python -m src index --max_chunk_size 2000 --mode hybrid
uv run python -m src search "How to configure the OpenAI server?" --k 5 --mode hybrid
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 --mode hybrid --save_directory data/output/search_results/UnansweredQuestions
```

## 📎 Resources

Bonus-specific, on top of `main`'s list:

- [Sentence-Transformers](https://www.sbert.net/) documentation — `all-MiniLM-L6-v2`, bi-encoder embeddings.
- Cormack, Clarke & Buettcher (2009), [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the RRF formula used for hybrid fusion.

### AI usage disclosure

Same workflow as `main` (see that README for the mandatory-phase breakdown): explain
rationale before/while writing, one module at a time, measure before/after any
retrieval-affecting change. Bonus-specific:

- **Semantic embeddings + hybrid RRF (#1/#2):** AI drafted the initial
  `embeddings.py`/`fuse()` implementation; a manual review found and fixed several bugs
  (typos, wrong return types, flake8/mypy violations) before it ran. When hybrid first
  underperformed pure lexical retrieval, AI helped isolate the root cause (the embedding
  model's 256-token cap silently truncating our ~2000-char chunks, and the lack of a
  path-token equivalent on the semantic side) through targeted experiments, each
  measured independently, rather than guessing at a fix.
- **Caching (#4):** AI proposed keying the cache on `(query, k, mode)` with an
  index-fingerprint invalidation scheme, then implemented and measured it (cold vs. warm
  timings above), confirming cold/warm outputs are byte-identical before accepting it.
- **Git history restructuring:** splitting bonus work onto this branch (keeping `main`
  lexical-only) was done via a non-destructive `git revert` at the user's direction.

Every generated block was read, questioned, and where necessary corrected before being
accepted — see [Challenges Faced](#-challenges-faced) for a concrete example (the hybrid
regression) where the first AI-proposed fix was tested, measured, and reverted before the
second was tried.
