*This project has been created as part of the 42 curriculum by ydimitra.*

# RAG Against the Machine

![42](https://img.shields.io/badge/42-Codam-000000?style=flat)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9?style=flat)
![bm25s](https://img.shields.io/badge/retrieval-bm25s-blue?style=flat)
![Qwen3](https://img.shields.io/badge/generation-Qwen3--0.6B-orange?style=flat)

*Tuning BM25 to find the right ~2000 characters out of 28,246 chunks of the vLLM codebase — then asking a 0.6B-parameter model to explain what it found.*

## 📑 Table of Contents
- [What is this project?](#-what-is-this-project)
- [Background — RAG, BM25, and the vLLM corpus](#-background--rag-bm25-and-the-vllm-corpus)
- [Project Structure](#-project-structure)
- [Reading Order — Where to Start](#-reading-order--where-to-start)
- [Documentation Index](#-documentation-index)
- [Installation](#-installation)
- [Usage](#-usage)
- [The Makefile](#-the-makefile)
- [Algorithm Explanation — BM25 Retrieval & Chunking](#-algorithm-explanation--bm25-retrieval--chunking)
- [Design Decisions](#-design-decisions)
- [Bonus Features Implemented](#-bonus-features-implemented)
- [Performance Analysis](#-performance-analysis)
- [Challenges Faced](#-challenges-faced)
- [Testing Strategy](#-testing-strategy)
- [Example Usage](#-example-usage)
- [Resources](#-resources)

## 🧩 What is this project?

**RAG Against the Machine** is a Retrieval-Augmented Generation system built over the
[vLLM 0.10.1](https://github.com/vllm-project/vllm) codebase (~2,800 files, docs +
Python source). Given a natural-language question such as:

> "How to configure the OpenAI server?"

the system does not ask the model to answer from memory — a 500M-parameter model has no
reliable internal knowledge of one specific version of a fast-moving open-source project.
Instead it retrieves the actual source locations the answer lives in and grounds
generation in them:

```json
{
  "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 1240,
  "last_character_index": 2850
}
```

The pipeline: **index** the corpus into ~28,246 chunks → **retrieve** the top-k most
relevant spans for a question via BM25 → **augment** the model's prompt with those spans
→ **generate** a grounded answer with `Qwen/Qwen3-0.6B` → **evaluate** recall@k
independently, by span overlap against a reference dataset, never by asking the model
whether its own answer "sounds right."

That last point matters: retrieval is graded on its own, so it has to be correct before
generation ever touches it — the twist here isn't the generation step, it's making a
26-line BM25 ranking function reliably land in the right ~2000-character window of the
right file.

## 🔍 Background — RAG, BM25, and the vLLM corpus

A few concepts worth knowing before diving into the code (all covered in more depth in
[Algorithm Explanation](#-algorithm-explanation--bm25-retrieval--chunking) and
[Design Decisions](#-design-decisions)):

- **RAG (Retrieval-Augmented Generation)** — instead of relying on a model's parametric
  memory, a retriever fetches relevant text from an external corpus at query time and
  inserts it into the prompt as grounding context before generation. Cuts hallucination
  on domain knowledge the model was never trained deeply on.
- **BM25** — a bag-of-words ranking function that scores a chunk for a query as
  `Σ idf(t) · tf-saturation(t, chunk)` over shared terms, controlled by `k1` (how fast
  term frequency saturates) and `b` (how strongly chunk length is normalized against).
  This project uses the [`bm25s`](https://github.com/xhluca/bm25s) library for scoring,
  fed by *our own* identifier-aware tokenizer.
- **Chunk** — the unit of retrieval: one `(file_path, first_character_index,
  last_character_index)` span, never a whole file. The corpus becomes ~28,246 focused
  chunks, each capped at 2000 characters.
- **Tokenizer** — not a subword/BPE tokenizer like an LLM uses. `tokenizer.py` lowercases,
  splits on non-alphanumerics, and emits identifiers *both* whole and as subtokens
  (`enable_lora` → `enable_lora`, `enable`, `lora`), so both a whole-word and a
  code-identifier query can match. The same tokenizer runs at index time and query time.
- **IoU (Intersection over Union)** — the metric `evaluate` (and the moulinette) use to
  decide whether a retrieved span "hits" a reference span: overlap length divided by
  union length, with the match bar set low (`> 0.05`) since being in the right file with
  a roughly right region is what's graded, not exact span precision.
- **`Qwen/Qwen3-0.6B`** — the small instruction-tuned model used for generation. It ships
  a "thinking" chat template that emits a `<think>` reasoning block by default; this
  project disables it (`enable_thinking=False`) purely for prompt-size and speed reasons
  — it is not what makes generation deterministic, that's the sampling settings in
  `generator.py`.

## 🗂 Project Structure

```
rag-against-machine/
├── src/            # CLI + every pipeline module (see below)
├── tests/          # pytest suite (see below)
├── docs/           # local-only design-decisions log (see below)
├── data/           # gitignored, populated locally (see below)
├── en_subject.pdf  # project subject
├── roadmap.md      # phase-by-phase build plan
├── Makefile
└── pyproject.toml  uv.lock
```

Click a folder to see what's inside it:

<details>
<summary><strong>📁 src/</strong> — CLI + every pipeline module</summary>

| file | purpose |
|---|---|
| `__main__.py` | Fire CLI class — thin, delegates to the modules below |
| `models.py` | pydantic data models + the span validator (rejects an over-2000-char `MinimalSource` at construction) |
| `tokenizer.py` | identifier-aware tokenization — same code path for indexing and querying |
| `chunking.py` | AST-based Python chunker + header-based Markdown chunker |
| `indexer.py` | builds the inverted index (`bm25s`) from chunks + tokens, persists it |
| `retriever.py` | `top_k()` / `search_dataset()` — BM25 scores → ranked `MinimalSource`s, mode dispatch |
| `generator.py` | grounds a prompt in retrieved sources, generates with `Qwen/Qwen3-0.6B` |
| `evaluator.py` | recall@k via `interval_iou()`, independent of the moulinette |

</details>

<details>
<summary><strong>📁 tests/</strong> — pytest suite (not graded, kept for defense confidence)</summary>

| file | purpose |
|---|---|
| `test_chunking.py` | offset round-trip invariant on real corpus files, AST edge cases |
| `test_tokenizer.py` | tokenizer contract — lowercasing, subtokens, 1-char/numeric dropping |
| `test_indexer.py` | index build/persist round-trip |
| `test_retriever.py` | `top_k()` ranking, `(score, -chunk_id)` tie-breaking |
| `test_evaluator.py` | `interval_iou()` edge cases, recall@k aggregation |
| `test_models.py` | pydantic validators, the impossible-to-construct oversized span |
| `test_cli.py` | CLI boundary never raises — empty query, gibberish, `k=0`, missing/malformed file |

Run with `make test`.

</details>

<details>
<summary><strong>📁 docs/</strong> — local-only design-decisions log</summary>

Gitignored by design (see [Documentation Index](#-documentation-index)) — the subject's
graded layout has no `docs/` folder, so this isn't part of what's submitted. Kept
locally as a running choice/alternative/why log per phase, feeding this README's
[Design Decisions](#-design-decisions) and [Challenges Faced](#-challenges-faced)
sections and my defense prep.

</details>

<details>
<summary><strong>📁 data/</strong> — gitignored, populated locally, never committed</summary>

```
data/
├── raw/vllm-0.10.1/                                  # the corpus to index
├── processed/                                        # persisted index (+ bonus caches)
├── datasets/{UnansweredQuestions,AnsweredQuestions}/  # question sets
└── output/
    ├── search_results/<DatasetScope>/
    └── search_results_and_answer/<DatasetScope>/
```

Every path here is a CLI argument (`--data_directory`, `--save_directory`, ...) — never
hard-coded, per the subject's hard constraints.

</details>

## 📖 Reading Order — Where to Start

If you're reviewing this project for the first time, this is the recommended order:

1. **`models.py`** — the pydantic data shapes (`MinimalSource`, `RagDataset`,
   `StudentSearchResults`, ...) everything else is built on, including the validator
   that makes an over-2000-char span impossible to construct.
2. **`tokenizer.py`** — how text becomes terms; the tokenizer contract that both indexing
   and querying share.
3. **`chunking.py`** — how a raw file becomes a list of `(file_path, first, last)` spans;
   the AST walk for Python, the header-based splitter for Markdown.
4. **`indexer.py`** — how chunks + tokens become a persisted `bm25s` index.
5. **`retriever.py`** — `top_k()` / `search_dataset()`, where BM25 scores turn into
   ranked `MinimalSource`s.
6. **`generator.py`** — how retrieved sources become a grounded prompt and an answer.
7. **`evaluator.py`** — recall@k, `interval_iou()`, and how a source counts as "found."
8. **`__main__.py`** — the thin CLI layer tying every phase together.

## 📚 Documentation Index

The subject's project layout (see [Project Structure](#-project-structure)) has no
`docs/` folder in the graded tree, so this README — not a set of separate per-topic
files — is the single source of truth a reviewer needs. A `docs/decisions.md` log is
kept locally for defense prep, but it is gitignored by design (the polished, final
version of every entry lives here instead). Use this table to jump straight to a topic:

| Topic | Covered in |
|---|---|
| Data models & span validation | [Design Decisions](#-design-decisions), `models.py` |
| Chunking algorithm (AST + Markdown) | [Algorithm Explanation](#-algorithm-explanation--bm25-retrieval--chunking) |
| Tokenizer contract & path-token indexing | [Background](#-background--rag-bm25-and-the-vllm-corpus), [Algorithm Explanation](#-algorithm-explanation--bm25-retrieval--chunking) |
| BM25 params & `bm25s` wiring | [Algorithm Explanation](#-algorithm-explanation--bm25-retrieval--chunking), [Performance Analysis](#-performance-analysis) |
| Query stopword stripping | [Design Decisions](#-design-decisions), [Challenges Faced](#-challenges-faced) |
| Semantic embeddings / hybrid RRF / caching | [Bonus Features Implemented](#-bonus-features-implemented) |
| CLI commands & flags | [Usage](#-usage) |

## ⚙️ Installation

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make install     # uv sync — installs bm25s, fire, pydantic, torch, transformers, tqdm
```

Place the vLLM corpus under `data/raw/vllm-0.10.1/` and the question datasets under
`data/datasets/{UnansweredQuestions,AnsweredQuestions}/` before indexing (see
[Example Usage](#-example-usage)). `data/` is gitignored — nothing under it is
committed, and no path in this project is ever hard-coded; every input/output location
is a CLI argument.

## ▶️ Usage

```bash
uv run python -m src <command> [options]
```

| command | purpose |
|---|---|
| `index --max_chunk_size <int>` | chunk `data/raw/`, build the persisted index |
| `search <query> --k <int>` | top-k sources for one query |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | batch retrieval over a dataset |
| `answer <query> --k <int>` | retrieve + generate an answer for one query |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | generate answers for saved search results |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | recall@k against a reference (own reimplementation only — never the moulinette) |

See every flag at any time with:

```bash
uv run python -m src --help
```

## 🛠 The Makefile

| command | what it does |
|---|---|
| `make install` | `uv sync` — installs all project dependencies |
| `make run` | Runs `uv run python -m src` (Fire's own `--help` on no args) |
| `make debug` | Runs the CLI under Python's `pdb` debugger |
| `make lint` | `flake8` + `mypy` with the mandatory flag set |
| `make lint-strict` | `mypy --strict --ignore-missing-imports` for stricter checking |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make test` | Runs the pytest suite (not graded, kept for the offset round-trip check) |

## 🧠 Algorithm Explanation — BM25 Retrieval & Chunking

**Chunking (`chunking.py`)**, dispatched by file extension:

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
  IoU > 0.05 bar against a reference span on its own. A file with no headers is one
  section spanning the whole file.
- Both funnel through `_split_span`, which enforces `max_chunk_size` (default 2000, the
  moulinette's hard `max_context_length` ceiling) by cutting at a blank line, then any
  newline, then mid-line as a last resort — the one code path every chunk passes through,
  so no chunk can ever exceed the cap regardless of which chunker produced it.
- **Offsets are ground truth.** Every file is read once as a single UTF-8 string; every
  chunk's `(first, last)` indexes into that exact string, so
  `open(file_path, encoding="utf-8").read()[first:last] == chunk.text` always holds. A
  pytest asserts this round-trip on the real corpus before any chunking change is kept.

**Retrieval (`retriever.py` / `tokenizer.py`):** the corpus is tokenized by the
identifier-aware tokenizer described in [Background](#-background--rag-bm25-and-the-vllm-corpus).
Two levers matter beyond the tokenizer itself:

- **Path-token indexing:** each chunk also gets the tokenized, corpus-relative file path
  appended to its token list, so a question naming a module (`gpu_model_runner.py`)
  can match that module's chunks even when the filename never appears in the code
  itself. Single biggest measured lever: code recall@5 **0.667 → 0.758** (see
  [Challenges Faced](#-challenges-faced)).
- **BM25 params `k1=1.3, b=0.85`**, baked in at index time by `bm25s`
  (`method="lucene"`, floored idf). Chosen by grid search over
  `k1∈{1.0,1.2,1.3,1.5}, b∈{0.75,0.85,0.9,1.0}` with path-token indexing and chunking
  fixed.

`bm25s` is fed our pre-tokenized lists (never raw text — that would silently drop
subtokens and path tokens), and results are re-sorted by `(score, -chunk_id)` for
deterministic ties.

## 🧭 Design Decisions

- **Chunk text is never stored in the index** — only `(file_path, first, last)`. Every
  consumer (generation, display) re-slices the corpus file, so spans are byte-identical
  by construction and the index stays small (~650KB pickle for 28k chunks).

- **A standard English stopword list, not one fitted to this corpus.** A
  document-frequency threshold scored marginally better on one dataset but needs a
  constant tuned against the very datasets used to evaluate it. The ordinary English list
  has no fitted parameter, transfers to any corpus, and measured at least as well
  everywhere — a better trade than one point bought with a magic number.

- **Stopwords are stripped from queries, never from the index.** Removing them from the
  index would change every chunk's length `dl` and all remaining `idf` values,
  invalidating the tuned `k1=1.3, b=0.85`. Filtering only the query changes which terms
  are asked about and nothing else, so no re-tuning is needed and no index format
  changes.

- **A query made entirely of stopwords keeps them.** A degraded ranking is better than
  returning nothing, so `strip_stopwords` falls back to the unfiltered terms in that
  case.

## 🎁 Bonus Features Implemented

Semantic embeddings, hybrid RRF fusion, and caching were all built and measured on the
`semantic-hybrid` branch (kept separate from `main` — hybrid measurably underperforms
the tuned mandatory lexical retriever, see numbers below, and `main` must stay the
validated mandatory baseline).

| Feature | Status | Where |
|---|---|---|
| Semantic embeddings (bonus #1) | ✅ built & measured | `semantic-hybrid` branch, `src/embeddings.py` |
| Hybrid retrieval / RRF fusion (bonus #2) | ✅ built & measured | `semantic-hybrid` branch, `retriever.fuse()` |
| Caching — index + query-results (bonus #4) | ✅ built & measured | `semantic-hybrid` branch, `retriever.py` cache layer |

### ✅ How to verify

```bash
git checkout semantic-hybrid
uv run python -m src search "..." --k 5 --mode hybrid
```

**Semantic embeddings** — `sentence-transformers`, `all-MiniLM-L6-v2`, CPU, 384-dim rows
L2-normalized at encode time. Chunk text is prefixed with its file path before embedding
(the semantic-side equivalent of path-token indexing), which alone lifted semantic
recall@5 from docs 0.53→0.58 and code 0.3737→0.5152. Full-corpus embed: **296.8s** for
28,246 chunks — under the 5-minute combined budget alongside the ~4s lexical build.

**Hybrid fusion** — Reciprocal Rank Fusion, `c=60`, top-100 candidates per retriever,
summing `1/(c + rank)` per retriever a chunk appears in. `mode` defaults to `"lexical"`
so the mandatory path is byte-identical to before this change; `--mode hybrid` is
opt-in.

Measured recall@5 (both public datasets):

| mode | docs | code |
|---|---|---|
| lexical | 0.8200 | 0.7576 |
| semantic | 0.5800 | 0.5152 |
| hybrid | 0.7800 | 0.7273 |

Hybrid is correctly implemented per spec but underperforms pure lexical on both
datasets — fusing in a much weaker retriever at equal rank-weight drags the strong one
down instead of complementing it, since path-token signal now overlaps between the two
retrievers rather than being a lexical exclusive.

**Caching** — two independent caches per subject ch. IX #4: an index cache
(`bm25s.BM25.load(..., mmap=True)`) and a `sqlite3` query-results cache keyed on
`sha256(mode, k, query)`, fingerprinted on the index file's mtime+size so a re-index
invalidates stale rows automatically. The real cold-start cost turned out to be loading
the `SentenceTransformer` model itself (**4.3s**), not the index — so the cache's value
for semantic/hybrid modes is skipping the model load entirely on a hit:

| run | cold | warm | speedup |
|---|---|---|---|
| single query (`search`) | 9.27s | 0.38s | ~24x |
| `search_dataset`, 100 questions | 12.39s | 0.40s | ~31x |

Output verified byte-identical cold vs warm via `diff` — caching changes speed only,
never results.

## 📊 Performance Analysis

Measured with the real CLI against both public datasets
(`data/datasets/*_public.json`, k=5, IoU > 0.05 match bar):

| | docs recall@5 | code recall@5 | mandatory bar |
|---|---|---|---|
| **lexical** | **0.8200** (82/100) | **0.7576** (75/99) | 0.80 / 0.50 ✅ |

**Query stopword removal** (see [Challenges Faced](#-challenges-faced) for the
mechanism). Measured on a second, harder pair of datasets as well, since the public sets
alone were not discriminating between these options:

| | docs recall@5 | code recall@5 | docs (harder set) | code (harder set) |
|---|---|---|---|---|
| without stopwords | 0.8200 | 0.7576 | 0.7900 | 0.7400 |
| **with stopwords** | 0.8200 | 0.7576 | **0.8100** | **0.7600** |

The gain shows up where it matters — on the harder sets, both docs and code improve, and
docs crosses the 0.80 bar. Public numbers are unchanged, so nothing is traded away.

**Chunk-size effect** (fixed path-token indexing, sweeping `max_chunk_size` /
`MIN_SECTION_SIZE`):

| max_chunk_size / MIN_SECTION_SIZE | docs recall@5 | code recall@5 |
|---|---|---|
| 1000 / 200 | 0.81 | 0.626 |
| **2000 / 600 (kept)** | **0.82** | **0.758** |

Smaller chunks helped docs marginally but cost 13 points of code recall — code chunks
that get split mid-function fall below the IoU > 0.05 bar against the reference span
more often, since the reference span is usually the whole function. `2000/600` is the
tuned default.

**Timings** (whole vLLM corpus, 28,246 chunks, real CLI runs):

| operation | time | budget |
|---|---|---|
| index | ~4.3s | ≤ 5 min |
| retrieval, 100 questions (bm25s scoring) | ~0.5s | ≤ 90s / 200 q |

## 🧩 Challenges Faced

**Code recall was stuck at 0.667 until a miss audit found the real pattern.** A miss
audit on early misses (questions naming a file directly, e.g. *"What does
`gpu_model_runner.py`'s `execute_model` do?"*) pointed at the root cause:

- Code chunk (function body) rarely contains its own filename as text
- Query naming that file → BM25 needs shared tokens, finds none → miss
- Fix: tokenize the file path too, attach those tokens to every chunk from that file
  (on top of the chunk's own code tokens)
- Now filename-mentioning queries have something to match against

This ended up being the single biggest lever measured across the whole project: code
recall@5 **0.667 → 0.758**, a bigger jump than the `k1`/`b` grid search or the
chunk-size sweep combined.

**Docs recall was capped by common words deciding the ranking.** Docs questions kept
losing to `.py` files, tests, benchmarks and shell scripts. A depth curve showed the
problem was *ranking*, not retrieval: recall@5 was 0.79 but recall@50 was 0.90, so
two-thirds of the misses were retrieved and merely ranked too low.

Decomposing one miss per query term made the cause exact. For *"What Python versions are
supported for vLLM CPU installation?"*, the correct file ranked 6th:

| term | wrong chunk (rank 1) | correct chunk (rank 6) |
|---|---|---|
| python | 3.156 | 2.570 |
| versions | 3.800 | 0.000 |
| are | 1.885 | 0.733 |
| **for** | **0.690** | **0.000** |
| cpu | 0.000 | 2.698 |
| installation | 2.500 | 4.757 |
| **total** | **12.031** | **11.369** |

The correct answer lost by 0.662 points, and the preposition **"for"** alone accounted
for 0.690 of that. The mechanism: BM25 scores a document as
`Σ idf(t) · tf-saturation(t,d)`, so *every* query term contributes a document-dependent
amount and acts as a tiebreaker. A widespread term cannot separate chunks by relevance,
but it still rewards whichever chunk repeats it most — and long source files repeat
`self`, `def`, `model` and `return` constantly, collecting near-maximum credit on every
single query.

The obvious objection is that `idf` should already handle this. It only handles the
extreme: `vllm` occurs in 100% of chunks and gets idf of exactly 0.000. The damage comes
from the mid-frequency band, which idf discounts but does not erase — `the` (36.9% of
chunks) still scores idf 0.998 and `self` (36.3%) scores 1.014, each worth up to
~2.3 points once term frequency saturates, against observed rank gaps of 0.66–4.7.

Fix: strip standard English stopwords **from the query only**. Removing them from the
index instead would change every chunk's length and all remaining idf values,
invalidating the tuned `k1`/`b`; filtering the query changes only which terms are asked
about. Punctuation and single characters needed no work — the tokenizer already matches
`[A-Za-z0-9_]+` and drops 1-character tokens, so commas and `a` never become terms.

Result: on the harder datasets, docs recall@5 **0.7900 → 0.8100** (clearing the 0.80
bar) and code **0.7400 → 0.7600**, with the public numbers unchanged at 0.8200 / 0.7576.

**Further improvement worth exploring:** a **corpus-derived** stoplist (terms in more
than 16% of chunks) also catches domain stopwords an English list misses — `py`, `def`,
`self`, `torch`, `model`.

## 🧪 Testing Strategy

`tests/` (pytest, not graded, kept for defense confidence):

- `test_chunking.py` — the offset round-trip invariant on real corpus files
  (`text[first:last] == chunk.text`), the one test that must always pass before any
  chunking change is merged; AST edge cases (decorators, oversized classes,
  `SyntaxError` fallback).
- `test_tokenizer.py` — the tokenizer contract: lowercasing, subtoken splitting,
  1-char/numeric dropping, identical behavior at index and query time.
- `test_indexer.py` — index build/persist round-trip.
- `test_retriever.py` — `top_k()` ranking, tie-breaking by `(score, -chunk_id)`.
- `test_evaluator.py` — `interval_iou()` edge cases (disjoint, contained, degenerate)
  and recall@k aggregation.
- `test_models.py` — pydantic validators, especially the impossible-to-construct
  over-2000-char `MinimalSource`.
- `test_cli.py` — CLI boundary never raises: empty query, gibberish, `k=0`, missing
  file, malformed JSON all produce a graceful message, never a traceback.

```bash
make test
```

## 💡 Example Usage

Dataset layout: `data/datasets/<DatasetScope>/dataset_{docs,code}_public.json` — an
`AnsweredQuestions` file carries ground-truth `sources` (use it for recall
evaluation); an `UnansweredQuestions` file does not (use it only for answer generation).

### 1. Build the index

```bash
uv run python -m src index \
  --max_chunk_size 2000 \
  --data_directory data/raw \
  --save_directory data/processed
```

### 2. Search a dataset (retrieval for every question)

Output filename mirrors the input basename — run once per file (docs, code).

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 3. Evaluate recall@k against a reference

Both args must point at the *same* dataset (code↔code, docs↔docs), and the reference
must be `AnsweredQuestions` (it needs `sources`):

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### One-off commands

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## 📎 Resources

- Robertson & Zaramba, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 scoring, `k1`/`b`.
- [`bm25s`](https://github.com/xhluca/bm25s) documentation — the BM25 library used on the graded path.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat template, `enable_thinking` flag.
- Python [`ast`](https://docs.python.org/3/library/ast.html) module docs — used for structure-aware Python chunking.
- [Pydantic v2](https://docs.pydantic.dev/latest/) docs — data model validation.
- [`uv`](https://docs.astral.sh/uv/) docs — dependency/project management.
- [sentence-transformers](https://www.sbert.net/) docs — `all-MiniLM-L6-v2`, used on the bonus `semantic-hybrid` branch.
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion* — the RRF formula used for hybrid fusion.

### AI usage disclosure

Claude Code was used as a pair-programming assistant throughout, following a fixed
workflow: explain the rationale for any non-obvious block (AST walking, BM25 scoring,
offset math) in chat before writing it, one phase or module at a time, and measure
recall@k before/after any retrieval-affecting change rather than tuning blind.

- **Chunking, tokenizer, indexer, retriever:** built phase by phase with AI
  pair-programming; every retrieval-affecting parameter (chunk size, `MIN_SECTION_SIZE`,
  `k1`/`b`, path-token indexing) was measured on the public datasets before being kept,
  with results logged as choice/alternative/why entries.
- **This README:** drafted by AI from the project's own commit history, code, and the
  recall numbers measured during development; reviewed and corrected by the author.

Every generated block was read, questioned, and where necessary corrected before being
accepted.
