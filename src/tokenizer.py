"""Identifier-aware tokenizer shared by indexing and querying.

The exact same function must tokenize corpus chunks and queries —
any asymmetry silently breaks retrieval. Identifiers are kept whole
AND split into subtokens (``enable_lora`` -> ``enable_lora``,
``enable``, ``lora``): questions either quote identifiers verbatim
(whole token matches) or paraphrase them (subtokens match). This
dual emission is the main lever for the code-recall target.
"""

import re

#: A word: a maximal run of letters, digits and underscores.
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

#: CamelCase splitter: ``HTTPServer`` -> ``HTTP``, ``Server``;
#: ``enableLora`` -> ``enable``, ``Lora``; digits split off.
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for indexing or querying.

    Lowercases, splits on non-alphanumerics, and for every compound
    identifier emits the whole (lowercased) identifier plus its
    snake_case/CamelCase subtokens. Single-character and pure-numeric
    tokens are dropped as noise.

    Args:
        text: Raw chunk or query text.

    Returns:
        Tokens in occurrence order (duplicates kept — term frequency
        is meaningful to BM25).
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group()
        lowered = word.lower()
        if _keep(lowered):
            tokens.append(lowered)
        parts = [
            part
            for piece in word.split("_")
            for part in _CAMEL_RE.findall(piece)
        ]
        for part in parts:
            part_lower = part.lower()
            if part_lower != lowered and _keep(part_lower):
                tokens.append(part_lower)
    return tokens


def _keep(token: str) -> bool:
    """Filter out noise tokens.

    Args:
        token: Lowercased candidate token.

    Returns:
        False for single characters and pure numbers, True otherwise.
    """
    return len(token) > 1 and not token.isdigit()
