"""Grounded answer generation with Qwen3-0.6B.

Sources are re-sliced from the corpus files they point at (chunk text
is never stored in the index, only offsets — see indexer.py), assembled
into a numbered context block, and handed to the model behind a system
prompt that forbids answering from outside knowledge. Sources are kept
best-ranked first and dropped worst-ranked first when the assembled
context would exceed the token budget, since the retriever already
orders them by relevance.
"""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.chunking import read_text
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

#: Model named by the subject; must work, other models optional on top.
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

#: Token budget for the assembled source context. Leaves headroom under
#: Qwen3's context window for the system prompt, question, and the
#: generated answer itself.
MAX_CONTEXT_TOKENS = 3000

#: Generation length cap; answers are expected to be a few sentences.
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the information in the "
    "numbered sources below. Do not use outside knowledge. If the "
    "sources do not contain the answer, say so explicitly instead of "
    "guessing."
)

#: Placeholder answer when no source text could be gathered at all.
NO_SOURCES_ANSWER = "No sources were available to answer this question."


def load_model(
    model_name: str = DEFAULT_MODEL,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the generation tokenizer and model once.

    Args:
        model_name: Hugging Face model id to load.

    Returns:
        The tokenizer and model, ready for ``generate_answer``.

    Raises:
        OSError: If the model cannot be downloaded or found locally.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _source_block(source: MinimalSource) -> str | None:
    """Re-slice one source's text from its corpus file.

    Args:
        source: A retrieved span, verbatim ``file_path`` plus offsets.

    Returns:
        A labeled text block for the prompt, or None if the file is
        missing or undecodable (e.g. a stale search result).
    """
    text = read_text(Path(source.file_path))
    if text is None:
        return None
    span = text[source.first_character_index:source.last_character_index]
    return f"[{source.file_path}]\n{span}"


def _fit_sources(
    tokenizer: PreTrainedTokenizerBase,
    sources: list[MinimalSource],
    max_context_tokens: int,
) -> list[str]:
    """Select source blocks that fit the context token budget.

    Args:
        tokenizer: Tokenizer used to count tokens (must match the
            generation model, since token counts are model-specific).
        sources: Retrieved sources, best-ranked first.
        max_context_tokens: Hard cap on the assembled context.

    Returns:
        Text blocks in ranked order, worst-ranked ones dropped first
        once the budget is exceeded. Always keeps at least the first
        available block, even if it alone exceeds the budget, so a
        question with sources never gets an empty context.
    """
    blocks: list[str] = []
    used_tokens = 0
    for source in sources:
        block = _source_block(source)
        if block is None:
            continue
        n_tokens = len(tokenizer.encode(block))
        if blocks and used_tokens + n_tokens > max_context_tokens:
            break
        blocks.append(block)
        used_tokens += n_tokens
    return blocks


def generate_answer(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    question: str,
    sources: list[MinimalSource],
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a grounded answer for one question from its sources.

    Args:
        tokenizer: Loaded Qwen3 tokenizer.
        model: Loaded Qwen3 causal LM.
        question: The question text.
        sources: Retrieved sources, best-ranked first.
        max_context_tokens: Token budget for the assembled context.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The model's answer text, or a fixed placeholder if no source
        text could be recovered at all.
    """
    blocks = _fit_sources(tokenizer, sources, max_context_tokens)
    if not blocks:
        return NO_SOURCES_ANSWER

    numbered = "\n\n".join(
        f"Source {i}:\n{block}" for i, block in enumerate(blocks, start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{numbered}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        # transformers' generate() stub resolves oddly against **inputs
        # unpacking; the call itself is the standard HF generation idiom.
        output_ids = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return decoded.strip()


def answer_results(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    results: StudentSearchResults,
    show_progress: bool = True,
) -> StudentSearchResultsAndAnswer:
    """Generate an answer for every question in saved search results.

    Args:
        tokenizer: Loaded Qwen3 tokenizer.
        model: Loaded Qwen3 causal LM.
        results: Previously saved retrieval results.
        show_progress: Display a tqdm bar over questions.

    Returns:
        StudentSearchResultsAndAnswer preserving question ids, order,
        and retrieved sources, with an answer added per question.
    """
    answers = []
    entries = tqdm(
        results.search_results,
        desc="Answering",
        unit="question",
        disable=not show_progress,
    )
    for entry in entries:
        answer_text = generate_answer(
            tokenizer, model, entry.question, entry.retrieved_sources
        )
        answers.append(
            MinimalAnswer(
                question_id=entry.question_id,
                question=entry.question,
                retrieved_sources=entry.retrieved_sources,
                answer=answer_text,
            )
        )
    return StudentSearchResultsAndAnswer(search_results=answers, k=results.k)


def save_answers(
    results: StudentSearchResultsAndAnswer, save_directory: Path, filename: str
) -> Path:
    """Write answered results as JSON into a directory.

    Args:
        results: The results to serialize.
        save_directory: Target directory (created if missing).
        filename: Output file name, conventionally the input's.

    Returns:
        Path of the written file.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    target = save_directory / filename
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))
    return target
