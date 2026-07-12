"""Smoke tests for the CLI skeleton."""

from src.__main__ import RagCLI


def test_cli_has_required_commands() -> None:
    """All six subject-mandated commands exist as callable methods."""
    cli = RagCLI()
    for command in (
        "index",
        "search",
        "search_dataset",
        "answer",
        "answer_dataset",
        "evaluate",
    ):
        assert callable(getattr(cli, command))
