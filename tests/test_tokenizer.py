"""Tokenizer contract tests — indexing and queries share this exact
behaviour, so these cases document the retrieval-critical rules.
"""

from src.tokenizer import tokenize


def test_snake_case_keeps_original_and_subtokens() -> None:
    """The subject's canonical example."""
    assert tokenize("enable_lora") == ["enable_lora", "enable", "lora"]


def test_camel_case_splits() -> None:
    """CamelCase yields whole identifier plus its words."""
    assert tokenize("OpenAIServingChat") == [
        "openaiservingchat", "open", "ai", "serving", "chat",
    ]


def test_acronym_boundary() -> None:
    """Acronym followed by a word splits at the right place."""
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_lowercase_and_nonalnum_split() -> None:
    """Plain sentences lowercase and split on punctuation; compound
    words still emit their subtokens (OpenAI -> open, ai)."""
    assert tokenize("How to configure the OpenAI server?") == [
        "how", "to", "configure", "the", "openai", "open", "ai", "server",
    ]


def test_leading_underscore_still_yields_word() -> None:
    """`_private` matches queries for `private`."""
    assert tokenize("_private") == ["_private", "private"]


def test_noise_dropped() -> None:
    """Single chars and pure numbers are filtered out."""
    assert tokenize("x = 5") == []
    assert tokenize("v0.10.1") == ["v0"]


def test_same_function_for_query_and_chunk() -> None:
    """A quoted identifier in a query hits the indexed identifier."""
    chunk_tokens = tokenize("def enable_lora(self) -> None:")
    query_tokens = tokenize('What does "enable_lora" do?')
    assert "enable_lora" in chunk_tokens
    assert "enable_lora" in query_tokens
