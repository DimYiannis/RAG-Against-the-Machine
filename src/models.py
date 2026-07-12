"""Pydantic data models mandated by the subject.

Field names match the subject verbatim; renaming any of them breaks the
moulinette comparison. Extra models/fields may be added later, renames
never. MinimalSource carries the one project-critical invariant: a span
may never exceed MAX_SOURCE_SPAN characters, because a single over-long
source invalidates an entire output file. The validator makes such an
instance impossible to construct.
"""

import uuid

from pydantic import BaseModel, Field, model_validator

MAX_SOURCE_SPAN = 2000


class MinimalSource(BaseModel):
    """A single retrieved span: file plus character offsets.

    Attributes:
        file_path: Path relative to project root, exactly as in the
            corpus (e.g. ``data/raw/vllm-0.10.1/...``); compared
            verbatim to the reference.
        first_character_index: Start offset into the file's text.
        last_character_index: End offset (exclusive) into the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def _check_span(self) -> "MinimalSource":
        """Reject negative, inverted, or over-long spans.

        Returns:
            The validated instance.

        Raises:
            ValueError: If offsets are negative, inverted, or the span
                exceeds MAX_SOURCE_SPAN characters.
        """
        first = self.first_character_index
        last = self.last_character_index
        if first < 0:
            raise ValueError(
                f"first_character_index must be >= 0, got {first}"
            )
        if last <= first:
            raise ValueError(
                f"span is empty or inverted: first={first}, last={last}"
            )
        if last - first > MAX_SOURCE_SPAN:
            raise ValueError(
                f"span {last - first} exceeds {MAX_SOURCE_SPAN} chars"
            )
        return self


class UnansweredQuestion(BaseModel):
    """A question without reference sources or answer.

    Attributes:
        question_id: Unique id; auto-generated uuid4 when absent.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer.

    Attributes:
        sources: Ground-truth spans the answer is based on.
        answer: The reference answer text.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not.

    Attributes:
        rag_questions: Questions; AnsweredQuestion is tried first in
            the union so entries with sources/answer parse as answered.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieval output for one question.

    Attributes:
        question_id: Id copied from the dataset question.
        question: The question text.
        retrieved_sources: Top-k spans returned by the retriever.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval output plus the generated answer for one question.

    Attributes:
        answer: Answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Full search output file: one entry per dataset question.

    Attributes:
        search_results: Per-question retrieval results.
        k: Number of results requested per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Full answer output file: one answered entry per question.

    Attributes:
        search_results: Per-question retrieval results with answers.
        k: Number of results requested per question.
    """

    search_results: list[MinimalAnswer]
    k: int
