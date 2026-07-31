"""
    identifier-aware tokenizer shared by indexing and querying.
 
    Identifiers are kept whole AND split into subtokens 
    -> (enable_lora -> enable_lora, enable, lora): 
    questions either quote whole token matches or paraphrase 
    them subtokens match. 
    This dual emission is the main lever for the code-recall target.
"""

# python's regex module
import re

# both compiled patterns precompiled at module load (re.compile) 
# so they're not recompiled per call, cheap to reuse across every chunk/query.

# matches a maximal run of letters/digits/underscore
WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# four alternatives, tried in order, used to split a single identifier piece into
#  CamelCase/acronym-aware subwords:
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

def tokenize(text:str) -> list[str]:
    """
        tokenize text for indexing, querying

        used the compiled patterns to split properly

        args: text

        return:
            tokens in occurence order
    """
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        word = match.group()
        lowered = word.lower()
        if _keep(lowered):
            tokens.append(lowered)
        parts = [
            part
            # underscore fragments 'enable_HTTPServer'
            for piece in word.split("_") 
            # scan fragments for capitals, [enable, HTTPServer]
            for part in CAMEL_RE.findall(piece)
        ]
        for part in parts:
            part_lower = part.lower()
            if part_lower != lowered and _keep(part_lower):
                tokens.append(part_lower)
    return tokens

def _keep(token: str) -> bool:
    """
        filter out noise
        
        args: 
            token: lowercased candidate token
        
        return: 
            false for single chars and pure nums, true otherwise
    """
    return len(token) > 1 and not token.isdigit()
