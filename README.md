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
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat template, `enable_thinking` flag.
- Python [`ast`](https://docs.python.org/3/library/ast.html) module docs — used for structure-aware Python chunking.
- [Pydantic v2](https://docs.pydantic.dev/latest/) docs — data model validation.
- [`uv`](https://docs.astral.sh/uv/) docs — dependency/project management.

### AI usage

Claude Code was used as a pair-programming assistant throughout, following
a fixed workflow: explain the rationale for any non-obvious block (AST walking, BM25
scoring, offset math) in chat before writing it, one phase or module at a time,
and measure recall@k before/after any retrieval-affecting change rather than tuning
blind.

- **Chunking, tokenizer, indexer, retriever:** built phase by phase with AI
  pair-programming; every retrieval-affecting parameter (chunk size, `MIN_SECTION_SIZE`,
  `k1`/`b`, path-token indexing) was measured on the public datasets before being kept,
  with results logged as choice/alternative/why entries.
- **This README:** drafted by AI from the project's own commit
  history, code, and the recall numbers measured during development; reviewed and
  corrected by the author.


Every generated block was read, questioned, and where necessary corrected before being
accepted.

## System architecture

```
CLI (Fire)  src/__main__.py
   │
   ├── index ─────────► indexer.py  (chunking.py + tokenizer.py → bm25s.BM25 index)
   │
   ├── search /
   │   search_dataset ─► retriever.py  top_k() / search_dataset()  (bm25s.retrieve())
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
| `index --max_chunk_size <int>` | chunk `data/raw/`, build the persisted index |
| `search <query> --k <int>` | top-k sources for one query |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | batch retrieval over a dataset |
| `answer <query> --k <int>` | retrieve + generate an answer for one query |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | generate answers for saved search results |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | recall@k against a reference (own iteration only — never the moulinette) |

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

The corpus is tokenized by a custom identifier-aware tokenizer (`tokenizer.py`) —
lowercase, split on non-alphanumerics, `snake_case`/`CamelCase` identifiers emitted
*both* whole and as subtokens (`enable_lora` → `enable_lora`, `enable`, `lora`), 1-char
and pure-numeric tokens dropped. The same tokenizer runs at index time and query time.
Two levers matter beyond the tokenizer itself:

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

## Performance analysis

Measured with the real CLI against both public datasets
(`data/datasets/*_public.json`, k=5, IoU>0.05 match bar):

| | docs recall@5 | code recall@5 | mandatory bar |
|---|---|---|---|
| **lexical** | **0.8200** (82/100) | **0.7576** (75/99) | 0.80 / 0.50 ✅ |

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
| index | ~4.3s | ≤ 5 min |
| retrieval, 100 questions (bm25s scoring) | ~0.5s | ≤ 90s / 200 q |

## Design decisions

- **Chunk text is never stored in the index** — only `(file_path, first, last)`. Every
  consumer (generation, display) re-slices the corpus file, so spans are byte-identical
  by construction and the index stays small (~650KB pickle for 28k chunks).


## Challenges faced

**Code recall was stuck at 0.667 until a miss audit found the real pattern.** Early on,
code-question misses split roughly evenly between "wrong file" and "right file, wrong
span" — but the wrong-file misses were disproportionately questions that *name* a file
or module ("What does `gpu_model_runner.py`'s `execute_model` do?") rather than quoting
code content. The filename itself never appeared in any chunk's tokens, so no amount of
BM25 tuning on the existing index could fix it — the fix had to be at index time, not
query time. Appending the tokenized corpus-relative path to every chunk of that file
(path-token indexing) turned out to be the single biggest lever measured across the
whole project: code recall@5 0.667 → 0.758, bigger than any `k1`/`b` or chunk-size
change.

## Example usage

```bash
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
```
