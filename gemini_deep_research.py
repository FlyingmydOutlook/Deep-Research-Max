#!/usr/bin/env python3
"""Minimal Gemini Deep Research CLI backed by the Interactions API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any

DEFAULT_AGENT = "deep-research-pro-preview-12-2025"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_VERSION = "v1beta"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}


class GeminiDeepResearchError(RuntimeError):
    """Raised when the Gemini Deep Research request fails."""


@dataclass
class Citation:
    title: str
    url: str


@dataclass
class ResearchResult:
    task_id: str
    status: str
    report: str
    citations: list[Citation]
    search_queries: list[str]
    reasoning_steps: int
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "report": self.report,
            "citations": [asdict(citation) for citation in self.citations],
            "search_queries": self.search_queries,
            "reasoning_steps": self.reasoning_steps,
            "total_steps": self.total_steps,
        }


def build_interactions_url(
    base_url: str,
    api_version: str = DEFAULT_API_VERSION,
    interaction_id: str | None = None,
    action: str | None = None,
) -> str:
    url = f"{base_url.rstrip('/')}/{api_version}/interactions"
    if interaction_id:
        url = f"{url}/{interaction_id}"
    if action:
        url = f"{url}:{action}"
    return url


def resolve_api_key(explicit_api_key: str | None) -> str:
    api_key = explicit_api_key or os.getenv("GEMINI_API_KEY") or os.getenv(
        "GOOGLE_API_KEY"
    )
    if not api_key:
        raise GeminiDeepResearchError(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return api_key


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": {"message": body}}
        message = parsed.get("error", {}).get("message") or body or str(error)
        raise GeminiDeepResearchError(f"Gemini API error ({error.code}): {message}") from error
    except urllib.error.URLError as error:
        raise GeminiDeepResearchError(f"Network error: {error.reason}") from error


def create_interaction(
    prompt: str,
    api_key: str,
    *,
    agent: str,
    base_url: str,
    api_version: str,
    system_instruction: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent": agent,
        "input": prompt,
        "background": True,
        "store": True,
    }
    if system_instruction:
        payload["system_instruction"] = system_instruction
    return request_json(
        "POST",
        build_interactions_url(base_url, api_version),
        api_key,
        payload,
    )


def get_interaction(
    interaction_id: str,
    api_key: str,
    *,
    base_url: str,
    api_version: str,
) -> dict[str, Any]:
    return request_json(
        "GET",
        build_interactions_url(base_url, api_version, interaction_id),
        api_key,
    )


def cancel_interaction(
    interaction_id: str,
    api_key: str,
    *,
    base_url: str,
    api_version: str,
) -> dict[str, Any]:
    return request_json(
        "POST",
        build_interactions_url(base_url, api_version, interaction_id, "cancel"),
        api_key,
        {},
    )


def extract_result(interaction: dict[str, Any]) -> ResearchResult:
    status = interaction.get("status", "unknown")
    task_id = interaction.get("id", "")
    outputs = interaction.get("outputs") or []

    if status != "completed":
        raise GeminiDeepResearchError(extract_failure_message(interaction))

    source_lookup: dict[str, Citation] = {}
    search_queries: list[str] = []
    reasoning_steps = 0
    final_report = ""
    final_annotations: list[dict[str, Any]] = []

    for output in outputs:
        output_type = output.get("type")
        if output_type == "text" and output.get("text"):
            final_report = output["text"]
            final_annotations = output.get("annotations") or []
        elif output_type == "thought":
            reasoning_steps += 1
        elif output_type == "google_search_call":
            arguments = output.get("arguments") or {}
            queries = arguments.get("queries") or []
            search_queries.extend(query for query in queries if query)
        elif output_type in {"google_search_result", "url_context_result"}:
            for result in output.get("result") or []:
                url = result.get("url")
                if not url:
                    continue
                title = result.get("title") or url
                source_lookup[url] = Citation(title=title, url=url)

    citations: list[Citation] = []
    seen_urls: set[str] = set()

    for annotation in final_annotations:
        source = annotation.get("source")
        if not source or source in seen_urls:
            continue
        citation = source_lookup.get(source, Citation(title=source, url=source))
        citations.append(citation)
        seen_urls.add(citation.url)

    for citation in source_lookup.values():
        if citation.url in seen_urls:
            continue
        citations.append(citation)
        seen_urls.add(citation.url)

    return ResearchResult(
        task_id=task_id,
        status=status,
        report=final_report,
        citations=citations,
        search_queries=search_queries,
        reasoning_steps=reasoning_steps,
        total_steps=len(outputs),
    )


def extract_failure_message(interaction: dict[str, Any]) -> str:
    error = interaction.get("error") or interaction.get("model_extra", {}).get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if message:
            return f"Research task failed: {message}"
    if error:
        return f"Research task failed: {error}"

    for output in reversed(interaction.get("outputs") or []):
        if output.get("type") == "text" and output.get("text"):
            return f"Research task {interaction.get('status', 'unknown')}: {output['text']}"

    return (
        f"Research task {interaction.get('id', '<unknown>')} ended with status "
        f"{interaction.get('status', 'unknown')}"
    )


def wait_for_completion(
    interaction_id: str,
    api_key: str,
    *,
    base_url: str,
    api_version: str,
    timeout: int,
    poll_interval: int,
    cancel_on_timeout: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        interaction = get_interaction(
            interaction_id,
            api_key,
            base_url=base_url,
            api_version=api_version,
        )
        status = interaction.get("status", "unknown")

        if status == "completed":
            return interaction
        if status == "requires_action":
            raise GeminiDeepResearchError(
                "Research task requires client-side tool handling, which this script does not support."
            )
        if status in TERMINAL_STATUSES:
            raise GeminiDeepResearchError(extract_failure_message(interaction))

        time.sleep(poll_interval)

    if cancel_on_timeout:
        try:
            cancel_interaction(
                interaction_id,
                api_key,
                base_url=base_url,
                api_version=api_version,
            )
        except GeminiDeepResearchError:
            pass

    raise GeminiDeepResearchError(
        f"Research task {interaction_id} did not complete within {timeout} seconds."
    )


def render_text(result: ResearchResult) -> str:
    lines = [result.report.strip() or "(empty report)"]

    if result.search_queries:
        lines.extend(
            [
                "",
                "Search queries:",
                *[f"- {query}" for query in result.search_queries],
            ]
        )

    if result.citations:
        lines.extend(
            [
                "",
                "Citations:",
                *[
                    f"- {citation.title}: {citation.url}"
                    for citation in result.citations
                ],
            ]
        )

    lines.extend(
        [
            "",
            f"Task ID: {result.task_id}",
            f"Reasoning steps: {result.reasoning_steps}",
            f"Total outputs: {result.total_steps}",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Research prompt to send to Gemini.")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Gemini research agent id.")
    parser.add_argument("--api-key", help="Gemini API key. Defaults to GEMINI_API_KEY / GOOGLE_API_KEY.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Gemini API base URL.")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="Gemini API version.")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum wait time in seconds.")
    parser.add_argument("--poll-interval", type=int, default=15, help="Polling interval in seconds.")
    parser.add_argument("--system-instruction", help="Optional system instruction.")
    parser.add_argument("--json", action="store_true", help="Print the final result as JSON.")
    parser.add_argument(
        "--no-cancel-on-timeout",
        action="store_true",
        help="Do not try to cancel the interaction when timing out.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.timeout <= 0:
        raise GeminiDeepResearchError("--timeout must be greater than 0.")
    if args.poll_interval <= 0:
        raise GeminiDeepResearchError("--poll-interval must be greater than 0.")

    api_key = resolve_api_key(args.api_key)

    interaction = create_interaction(
        args.prompt,
        api_key,
        agent=args.agent,
        base_url=args.base_url,
        api_version=args.api_version,
        system_instruction=args.system_instruction,
    )

    interaction_id = interaction.get("id")
    if not interaction_id:
        raise GeminiDeepResearchError("Gemini API did not return an interaction id.")

    completed = wait_for_completion(
        interaction_id,
        api_key,
        base_url=args.base_url,
        api_version=args.api_version,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        cancel_on_timeout=not args.no_cancel_on_timeout,
    )
    result = extract_result(completed)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text(result))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GeminiDeepResearchError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
