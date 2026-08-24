"""
    grounded answer generation with Qwen3-0.6B.

    sources are re-sliced from the corpus files they point at (chunk text
    is never stored in the index, only offsets),
    assembled into a numbered context block, and handed to the model
    behind a system prompt that forbids answering from outside knowledge.

    Sources are kept best-ranked first and dropped worst-ranked first
    when the assembled context would exceed the token budget, since the
    retriever already orders them by relevance.

    chunk metadata -> reslice corpus -> context block -> prompt -> answer
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

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# token budget for the assembled source block
MAX_CONTEXT_TOKENS = 3000

# generation length cap
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the information in the "
    "numbered sources below. Do not use outside knowledge. If the "
    "sources do not contain the answer, say so explicitly instead of "
    "guessing."
)

# default answer when no source is gathered
NO_SOURCES_ANSWER = "No sources were available to answer this question."


def load_model(
    model_name: str = DEFAULT_MODEL,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
        load generation tokenizer and model

        args:
            model_name

        return:
            tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load model weights
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _source_block(source: MinimalSource) -> str | None:
    """
        re-slice one source's text from corpus file

        args:
            text block for the prompt, or none if
            file is missing
    """
    text = read_text(Path(source.file_path))
    if text is None:
        return None
    span = text[source.first_character_index:source.last_character_index]
    return f"[{source.file_path}]\n{span}"


def _fit_sources(
    tokenizer: PreTrainedTokenizerBase,
    sources: list[MinimalSource],
    max_control_tokens: int,
) -> list[str]:

    """
        select source blocks that fit the token budget

        args:
            tokenizer: tokenizer used to count tokens
                (must match the generation model, since token
                counts are model-specific)
            sources: retrieved sources, best ranked first
            max_context_tokens: cap on the assembled context

        return:
            text blocks in ranked order, worst ranked ones are dropped first
            once the budget is exceeded.
            always keep the first available block even if the budget
            is exceeded,
            so a question with sources never gets an empty context
    """
    blocks: list[str] = []
    used_tokens = 0
    for source in sources:
        block = _source_block(source)
        if block is None:
            continue
        n_tokens = len(tokenizer.encode(block))
        if blocks and used_tokens + n_tokens > max_control_tokens:
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
    """
        generate grounded answer for one question from its sources

        args:
            tokenizer: loaded Qwen3 tokenizer
            model: loaded Qwen3 casual LM
            question
            sources: retrieved sources, best-ranked first
            max_context_tokens
            max_new_tokens

        return:
            model's answer or fixed response if no source was recovered
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
    # avoid gradient tracking and backprop, only inference
    with torch.no_grad():
        # transformers' generate() stub resolves oddly against **inputs
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
    """
        generate an answer for every question in saved search results

        args:
            tokenizer
            model
            results: previously saved retrieval results
            show_progress: tqdm bar
        return:
            StudentSearchResultsAndAnswer
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
    """
        write answered results as json into a directory

        args:
            results: results to serialize
            save_directory
            filename

        return
            path of the written file
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    target = save_directory / filename
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))
    return target
