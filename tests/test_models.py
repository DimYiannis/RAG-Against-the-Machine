"""Round-trip and validator tests for the subject's data models."""

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)

# Subject VI.5 example, verbatim structure.
SUBJECT_EXAMPLE = """
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How to configure OpenAI server?",
      "retrieved_sources": [
        {
          "file_path":
            "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        },
        {
          "file_path":
            "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
          "first_character_index": 267,
          "last_character_index": 400
        }
      ]
    }
  ],
  "k": 10
}
"""


def test_subject_example_round_trips() -> None:
    """Subject's example JSON survives validate -> dump -> validate."""
    parsed = StudentSearchResults.model_validate_json(SUBJECT_EXAMPLE)
    dumped = parsed.model_dump_json()
    assert StudentSearchResults.model_validate_json(dumped) == parsed
    assert parsed.k == 10
    assert parsed.search_results[0].retrieved_sources[0].file_path == (
        "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md"
    )


def test_dataset_union_disambiguates_on_field_presence() -> None:
    """Entries with sources+answer parse answered; bare ones don't."""
    dataset = RagDataset.model_validate(
        {
            "rag_questions": [
                {"question_id": "a", "question": "answered?",
                 "sources": [], "answer": "yes"},
                {"question_id": "u", "question": "unanswered?"},
            ]
        }
    )
    answered, unanswered = dataset.rag_questions
    assert isinstance(answered, AnsweredQuestion)
    assert isinstance(unanswered, UnansweredQuestion)
    assert not isinstance(unanswered, AnsweredQuestion)


def test_question_id_defaults_to_uuid() -> None:
    """Missing question_id is auto-filled and unique."""
    first = UnansweredQuestion(question="q")
    second = UnansweredQuestion(question="q")
    assert first.question_id and second.question_id
    assert first.question_id != second.question_id


def test_span_at_limit_is_valid() -> None:
    """A span of exactly 2000 characters is allowed."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/README.md",
        first_character_index=0,
        last_character_index=2000,
    )
    assert source.last_character_index == 2000


@pytest.mark.parametrize(
    ("first", "last"),
    [
        (0, 2001),  # over the 2000-char limit
        (-1, 100),  # negative start
        (100, 100),  # empty span
        (200, 100),  # inverted span
    ],
)
def test_invalid_spans_cannot_be_constructed(first: int, last: int) -> None:
    """Validator rejects over-long, negative, empty, inverted spans."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="data/raw/vllm-0.10.1/README.md",
            first_character_index=first,
            last_character_index=last,
        )
