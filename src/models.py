"""
    pydantic models following the subject's guidelines.
    standard field names so that moulinnette wont crush

    based on subject max span must not exceed 2000 chars
"""
import uuid

from pydantic import BaseModel, Field, model_validator

MAX_SPAN = 2000

class MinimalSource(BaseModel):
    """
        a single retrieved span
        attrs:
            file_path
            first_char_index
            last_char_index
    """
    file_path: str
    first_char_index: int
    last_char_index: int

    @model_validator(mode="after")
    def _check_span(self) -> "MinimalSource":
        """
            check if span size is acceptable
            - negative
            - exceeding max
            - inverted (last char preceeds the first)
        """
        first = self.first_char_index
        last = self.last_char_index
        if first < 0:
            raise ValueError("first char index must be bigger than 0")
        if last <= first:
            raise ValueError(f"span is empty or inverted: first char={first}, last char={last}")
        if last - first > MAX_SPAN:
            raise ValueError(f"span {last - first} exceeds the maximum capacity a chunk can have")
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
