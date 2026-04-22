import unittest

from gemini_deep_research import (
    Citation,
    build_interactions_url,
    extract_failure_message,
    extract_result,
)


class GeminiDeepResearchTests(unittest.TestCase):
    def test_build_interactions_url(self) -> None:
        self.assertEqual(
            build_interactions_url("https://generativelanguage.googleapis.com"),
            "https://generativelanguage.googleapis.com/v1beta/interactions",
        )
        self.assertEqual(
            build_interactions_url(
                "https://generativelanguage.googleapis.com/",
                interaction_id="abc123",
                action="cancel",
            ),
            "https://generativelanguage.googleapis.com/v1beta/interactions/abc123:cancel",
        )

    def test_extract_result_collects_report_queries_and_citations(self) -> None:
        interaction = {
            "id": "task-1",
            "status": "completed",
            "outputs": [
                {"type": "thought", "text": "thinking"},
                {
                    "type": "google_search_call",
                    "arguments": {"queries": ["gemini deep research api"]},
                },
                {
                    "type": "google_search_result",
                    "result": [
                        {
                            "title": "Gemini Deep Research",
                            "url": "https://example.com/deep-research",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "Final report",
                    "annotations": [
                        {"source": "https://example.com/deep-research"},
                        {"source": "https://example.com/extra"},
                    ],
                },
            ],
        }

        result = extract_result(interaction)

        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(result.report, "Final report")
        self.assertEqual(result.search_queries, ["gemini deep research api"])
        self.assertEqual(result.reasoning_steps, 1)
        self.assertEqual(result.total_steps, 4)
        self.assertEqual(
            result.citations,
            [
                Citation(
                    title="Gemini Deep Research",
                    url="https://example.com/deep-research",
                ),
                Citation(
                    title="https://example.com/extra",
                    url="https://example.com/extra",
                ),
            ],
        )

    def test_extract_failure_message_prefers_api_error(self) -> None:
        message = extract_failure_message(
            {
                "id": "task-2",
                "status": "failed",
                "error": {"message": "permission denied"},
            }
        )

        self.assertEqual(message, "Research task failed: permission denied")


if __name__ == "__main__":
    unittest.main()
