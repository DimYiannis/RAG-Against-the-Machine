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
            first_character_index
            last_character_index
    """
    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def _check_span(self) -> "MinimalSource":
        """
            check if span size is acceptable
            - negative
            - exceeding max
            - inverted (last char preceeds the first)
        """
        first = self.first_character_index
        last = self.last_character_index
        if first < 0:
            raise ValueError("first char index must be bigger than 0")
        if last <= first:
            raise ValueError(
                f"span is empty or inverted: first char={first}, "
                f"last char={last}"
            )
        if last - first > MAX_SPAN:
            raise ValueError(
                f"span {last - first} exceeds the maximum capacity "
                "a chunk can have"
            )
        return self


class UnansweredQuestion(BaseModel):
    """
        represent an unanswered question

        attrs:
        question_id
        question
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
        represent an unanswered question with its reference sources and answer.

        attrs:
            sources
            answer
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
        dataset of RAG questions, answered or not.

        attrs:
        rag_questions:
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
        retrieval output for one question.

        attrs:
            question_id
            question
            retrieved_sources: Top-k spans returned by the retriever.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
        retrieval output plus the generated answer for one question.

        attrs:
            answer
    """

    answer: str


class StudentSearchResults(BaseModel):
    """
        full search output file: one entry per dataset question.

        attrs:
            search_results: Per-question retrieval results.
            k: number of results requested per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
        full answer output file: one answered entry per question.

        attrs:
            search_results: per-question retrieval results with answers.
            k: number of results requested per question.
    """

    search_results: list[MinimalAnswer]
    k: int
