*This project has been created as part of the 42 curriculum by ydimitra.*

# RAG Against the Machine

## Description

A Retrieval-Augmented Generation system over the [vLLM 0.10.1](https://github.com/vllm-project/vllm)
codebase (~2,800 files, docs + Python source). Given a natural-language question about
the codebase, the system retrieves the most relevant source locations (file + character
span) and generates a grounded answer with `Qwen/Qwen3-0.6B`. Retrieval quality is
measured with recall@k against a reference dataset of question/source pairs.

The pipeline: **index** the corpus → **retrieve** top-k spans for a question →
**augment** the model's context with those spans → **generate** an answer → **evaluate**
recall independently of the retrieval step.

Two retrieval families are implemented:

- **Mandatory (`main` branch):** lexical retrieval only — BM25 via the [`bm25s`](https://github.com/xhluca/bm25s)
  library, fed a custom identifier-aware tokenizer. This is the graded/mandatory system.
- **Bonus (`semantic-hybrid` branch):** adds dense embeddings (`all-MiniLM-L6-v2`),
  RRF fusion of lexical + semantic rankings, and a two-level cache. See
  [Bonus work](#bonus-work) below.

## Instructions

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make install                 # uv sync
make lint                    # flake8 + mypy (project's required flags)
make test                    # pytest (not graded, kept for the offset round-trip check)
```

Place the vLLM corpus under `data/raw/vllm-0.10.1/` and the question datasets under
`data/datasets/{UnansweredQuestions,AnsweredQuestions}/` before indexing (see
[Example usage](#example-usage)). `data/` is gitignored — nothing under it is committed.

Every command is `uv run python -m src <command> [options]`; see `make run` /
`make debug` (pdb) for the Makefile shortcuts, and `Command-Line Interface` in
[System architecture](#system-architecture) for the full command list.

## Resources

- Robertson & Zaramba, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 scoring, `k1`/`b`.
- [`bm25s`](https://github.com/xhluca/bm25s) documentation — the BM25 library used on the graded path.
- [Sentence-Transformers](https://www.sbert.net/) documentation — `all-MiniLM-L6-v2`, bi-encoder embeddings.
- Cormack, Clarke & Buettcher (2009), [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the RRF formula used for hybrid fusion.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat template, `enable_thinking` flag.
- Python [`ast`](https://docs.python.org/3/library/ast.html) module docs — used for structure-aware Python chunking.
- [Pydantic v2](https://docs.pydantic.dev/latest/) docs — data model validation.
- [`uv`](https://docs.astral.sh/uv/) docs — dependency/project management.

### AI usage

Claude Code (Anthropic) was used as a pair-programming assistant throughout, following
a fixed workflow: explain the rationale for any non-obvious block (AST walking, BM25
scoring, offset math, RRF) in chat before/while writing it, one phase or module at a
time, and measure recall@k before/after any retrieval-affecting change rather than
tuning blind. Breakdown by area:

- **Chunking, tokenizer, indexer, retriever (mandatory):** built phase by phase with AI
  pair-programming; every retrieval-affecting parameter (chunk size, `MIN_SECTION_SIZE`,
  `k1`/`b`, path-token indexing) was measured on the public datasets before being kept,
  with results logged as choice/alternative/why entries.
- **Semantic embeddings + hybrid RRF (bonus #1/#2):** AI drafted the initial
  `embeddings.py`/`fuse()` implementation; a manual review found and fixed several bugs
  (typos, wrong return types, flake8/mypy violations) before it ran. When hybrid first
  underperformed pure lexical retrieval, AI helped isolate the root cause (the embedding
  model's 256-token cap silently truncating our ~2000-char chunks, and the lack of a
  path-token equivalent on the semantic side) through targeted experiments, each
  measured independently, rather than guessing at a fix.
- **Caching (bonus #4):** AI proposed keying the cache on `(query, k, mode)` with an
  index-fingerprint invalidation scheme, then implemented and measured it (cold vs. warm
  timings below), confirming cold/warm outputs are byte-identical before accepting it.
- **Git history restructuring:** splitting bonus work onto a separate `semantic-hybrid`
  branch (keeping `main` lexical-only) was done via a non-destructive `git revert`
  (rather than a history rewrite) at the user's direction, to keep `main` minimal and
  safe to grade while preserving full bonus history elsewhere.
- **This README and `docs/decisions.md`:** drafted by AI from the project's own commit
  history, code, and the recall numbers measured during development; reviewed and
  corrected by the author.

Every generated block was read, questioned, and where necessary corrected before being
accepted — see the "Challenges faced" section below for a concrete example (the hybrid
regression) where the first AI-proposed fix was tested, measured, and reverted before
the second one was tried.

## System architecture

```
CLI (Fire)  src/__main__.py
   │
   ├── index ─────────► indexer.py  (chunking.py + tokenizer.py → bm25s.BM25 index)
   │                     [semantic-hybrid only] embeddings.py (SentenceTransformer → .npy matrix)
   │
   ├── search /
   │   search_dataset ─► retriever.py  top_k() / search_dataset()
   │                     mode="lexical"  → bm25s.retrieve()
   │                     mode="semantic" → embeddings.semantic_top_k() (cosine similarity)
   │                     mode="hybrid"   → both, fused by retriever.fuse() (RRF)
   │                     [semantic-hybrid only] cache.py checked first: (query,k,mode) → cached result
   │
   ├── answer /
   │   answer_dataset ─► retriever.top_k() → generator.py (Qwen/Qwen3-0.6B, grounded prompt)
   │
   └── evaluate ───────► evaluator.py  recall@k (IoU > 0.05), independent of the moulinette
```

`src/__main__.py` is intentionally thin: it validates nothing itself, wraps every
command in one try/except (unhandled tracebacks are a hard failure per the subject), and
delegates to the module for the corresponding phase. Data flows through pydantic models
(`src/models.py`) at every stage boundary — `MinimalSource`, `RagDataset`,
`StudentSearchResults`, etc. — exactly as specified, with a validator on `MinimalSource`
that makes constructing an out-of-bounds or over-2000-char span impossible rather than
merely discouraged.

**Command-Line Interface** (`uv run python -m src <command>`):

| command | purpose |
|---|---|
| `index --max_chunk_size <int> [--mode lexical\|semantic\|hybrid]` | chunk `data/raw/`, build the persisted index |
| `search <query> --k <int> [--mode ...]` | top-k sources for one query |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir> [--mode ...]` | batch retrieval over a dataset |
| `answer <query> --k <int> [--mode ...]` | retrieve + generate an answer for one query |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | generate answers for saved search results |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | recall@k against a reference (own iteration only — never the moulinette) |

`--mode` defaults to `lexical` everywhere; omitting it reproduces the mandatory
behaviour exactly. It only does anything on the `semantic-hybrid` branch — `main` has no
`--mode` flag at all, since it carries no code path that would use it.

## Chunking strategy

Two distinct chunkers, dispatched by file extension (`chunking.py`):

- **Python (`chunk_python`):** walks the file's AST; each top-level function or class
  becomes its own chunk (decorators included in the span), and the code *between*
  definitions (imports, module docstring, constants) becomes its own chunk too — nothing
  textual is dropped. A class larger than `max_chunk_size` recurses into its own body one
  level, so its oversized methods split out individually while the class header (the gap
  before the first method) becomes its own small chunk — one algorithm, no special
  casing. Files that fail to parse (`SyntaxError`) fall back to line-window chunking.
- **Markdown/text (`chunk_markdown`):** each ATX header (`#` … `######`) starts a new
  section, so a header always stays attached to its own body. Sections shorter than
  `MIN_SECTION_SIZE` (600 chars) are merged into the previous section, as long as the
  merge stays under `max_chunk_size` — a short section alone rarely clears the
  IoU>0.05 bar against a reference span on its own. A file with no headers is one section
  spanning the whole file.
- Both funnel through `_split_span`, which enforces `max_chunk_size` (default 2000, the
  moulinette's hard `max_context_length` ceiling) by cutting at a blank line, then any
  newline, then mid-line as a last resort — the one code path every chunk passes through,
  so no chunk can ever exceed the cap regardless of which chunker produced it.
- **Offsets are ground truth.** Every file is read once as a single UTF-8 string; every
  chunk's `(first, last)` indexes into that exact string, so
  `open(file_path, encoding="utf-8").read()[first:last] == chunk.text` always holds. A
  pytest asserts this round-trip on the real corpus before any chunking change is kept.

## Retrieval method

**Lexical (mandatory, `bm25s` library):** the corpus is tokenized by a custom
identifier-aware tokenizer (`tokenizer.py`) — lowercase, split on non-alphanumerics,
`snake_case`/`CamelCase` identifiers emitted *both* whole and as subtokens
(`enable_lora` → `enable_lora`, `enable`, `lora`), 1-char and pure-numeric tokens
dropped. The same tokenizer runs at index time and query time. Two levers matter beyond
the tokenizer itself:

- **Path-token indexing:** each chunk also gets the tokenized, corpus-relative file path
  appended to its token list, so a question naming a module (`gpu_model_runner.py`)
  can match that module's chunks even when the filename never appears in the code
  itself. Single biggest measured lever: code recall@5 **0.667 → 0.758**.
- **BM25 params `k1=1.3, b=0.85`**, baked in at index time by `bm25s` (`method="lucene"`,
  floored idf). Chosen by grid search over `k1∈{1.0,1.2,1.3,1.5}, b∈{0.75,0.85,0.9,1.0}`
  with path-token indexing and chunking fixed.

`bm25s` is fed our pre-tokenized lists (never raw text — that would silently drop
subtokens and path tokens), and results are re-sorted by `(score, -chunk_id)` for
deterministic ties.

**Semantic (`semantic-hybrid` branch, bonus #1):** every chunk is embedded with
`all-MiniLM-L6-v2` (384-dim, CPU), rows L2-normalized so cosine similarity reduces to a
single matrix-vector dot product at query time (`np.argpartition` for top-k, not a full
sort). Chunk text is prefixed with its file path before embedding — the semantic
equivalent of path-token indexing, since a bi-encoder has no other way to see a filename.

**Hybrid (`semantic-hybrid` branch, bonus #2):** Reciprocal Rank Fusion of the lexical
and semantic rankings — `RRF(d) = Σ 1/(c + rank_i(d))` over both retrievers, 1-based
ranks, `c=60` (Cormack et al. standard), top-100 candidates pulled from each side before
fusing. Rank-based fusion needs no cross-scale normalization between BM25 and cosine
scores.

## Performance analysis

Measured with the real CLI against both public datasets
(`data/datasets/*_public.json`, k=5, IoU>0.05 match bar):

| mode | docs recall@5 | code recall@5 | mandatory bar |
|---|---|---|---|
| **lexical** (`main`) | **0.8200** (82/100) | **0.7576** (75/99) | 0.80 / 0.50 ✅ |
| semantic (`semantic-hybrid`) | 0.5800 | 0.5152 | — bonus only |
| hybrid (`semantic-hybrid`) | 0.7800 | 0.7273 | — bonus only |

**Chunk-size effect** (fixed path-token indexing, sweeping `max_chunk_size` /
`MIN_SECTION_SIZE`):

| max_chunk_size / MIN_SECTION_SIZE | docs recall@5 | code recall@5 |
|---|---|---|
| 1000 / 200 | 0.81 | 0.626 |
| **2000 / 600 (kept)** | **0.82** | **0.758** |

Smaller chunks helped docs marginally but cost 13 points of code recall — code chunks
that get split mid-function fall below the IoU>0.05 bar against the reference span more
often, since the reference span is usually the whole function. `2000/600` is the tuned
default.

**Timings** (whole vLLM corpus, 28,246 chunks, real CLI runs):

| operation | time | budget |
|---|---|---|
| index (lexical only) | ~4.3s | ≤ 5 min |
| index (+ embeddings, `--mode semantic`/`hybrid`) | ~300s (296.8s embed step) | ≤ 5 min |
| retrieval, 100 questions (bm25s scoring) | ~0.5s | ≤ 90s / 200 q |
| single query, hybrid, cold cache | 9.27s | — |
| single query, hybrid, **warm cache** | **0.38s** (~24x) | — |
| `search_dataset`, 100 questions, hybrid, cold cache | 12.39s | — |
| `search_dataset`, 100 questions, hybrid, **warm cache** | **0.40s** (~31x) | — |

Warm-cache runs skip loading the embedding model entirely (100% cache hit) — output
verified byte-identical to the cold run via `diff`.

## Design decisions

The full choice/alternative/why log lives in `docs/decisions.md` (kept locally,
gitignored as defense-prep notes per the subject's "not merely described" bonus rule —
this section is its polished summary). Highlights:

- **`bm25s` library over a hand-rolled scorer** (`main`): keeps less scoring code on the
  graded path; the hand-rolled version is kept on a separate `bm25-handrolled` branch as
  reference — measured *identical* recall (0.8200/0.7576) once fed the same
  tokenizer output, confirming the tokenizer/chunker were always the differentiator, not
  the scoring loop.
- **Chunk text is never stored in the index** — only `(file_path, first, last)`. Every
  consumer (embeddings, generation, display) re-slices the corpus file, so spans are
  byte-identical by construction and the index stays small (~650KB pickle for 28k
  chunks).
- **Bonus work lives on a separate `semantic-hybrid` branch, not `main`.** `main` stays
  minimal and lexical-only — safe to grade, nothing to accidentally break the mandatory
  path. The split was done with `git revert` (not a history rewrite), so the full
  development history for the bonus work is preserved and reachable, just not on `main`.
- **Hybrid mode is kept even though it underperforms pure lexical** (0.78/0.7273 vs.
  0.82/0.7576) — see Challenges faced. It's correctly implemented per the subject's RRF
  spec and demonstrates the technique; not tuned further past the point of diminishing,
  unmeasured returns (the "no blind tuning" rule).
- **Caching keyed on `(query, k, mode)`, not `(query, k)`** — `mode` is a dimension we
  introduced ourselves (the subject only requires hybrid to combine both rankings, not
  that retrieval be mode-selectable), so a lexical result must never satisfy a hybrid
  cache lookup or vice versa. Invalidated by an index-file mtime+size fingerprint, so a
  reindex can never serve stale results.

## Challenges faced

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
demonstrable, just not a recall win on this corpus with this embedding model.

**Cache had to be wired around the model load, not around retrieval math.** The naive
place to cache is inside `top_k()` — but by the time `top_k()` is called, the embedding
model (the actual ~4.3s cold-start cost) has already been loaded. The cache check had to
move to the CLI/`search_dataset` boundary, *before* `_load_semantic()` runs, so a cache
hit can skip loading the model entirely rather than just skip re-scoring.

## Example usage

```bash
# mandatory (works on either branch)
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How to configure the OpenAI server?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# bonus (git checkout semantic-hybrid first)
uv run python -m src index --max_chunk_size 2000 --mode hybrid
uv run python -m src search "How to configure the OpenAI server?" --k 5 --mode hybrid
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 --mode hybrid --save_directory data/output/search_results/UnansweredQuestions
```

## Bonus work

Implemented and measured on the `semantic-hybrid` branch (`git checkout semantic-hybrid`):

- **#1 Semantic embeddings** — `src/embeddings.py`, `all-MiniLM-L6-v2`, CPU, L2-normalized
  vector matrix persisted alongside the lexical index.
- **#2 Hybrid retrieval** — `retriever.fuse()`, Reciprocal Rank Fusion (`c=60`,
  top-100/retriever) of the lexical and semantic rankings.
- **#4 Caching** — `src/cache.py`, two caches: `bm25s` mmap load for cold-start index
  loading, and a sqlite3 query-results cache keyed `(query, k, mode)`, invalidated by an
  index-file fingerprint. Measured ~24-31x speedup on repeated queries (see Performance
  analysis).

Not implemented: **#3 incremental indexing**, **#5 local HTTP API**.
