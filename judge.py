"""Classify consistency-check inconsistency items into MCP threat categories."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

PROMPT_THREAT_PATH = Path(__file__).parent / "prompt" / "judge_threat.md"
VALID_SECURITY_LABELS = {
    "instruction_injection",
    "misleading_description",
    "potentially_malicious_code",
    "unsafe_implementation",
}


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_prompt(
    template: str,
    inconsistency_items: str,
    developer_description: str,
    code_derived_description: str,
) -> str:
    inconsistency_token = f"__CDCHECKER_INCONSISTENCY_{uuid.uuid4().hex}__"
    dev_token = f"__CDCHECKER_DEV_DESC_{uuid.uuid4().hex}__"
    code_token = f"__CDCHECKER_CODE_DESC_{uuid.uuid4().hex}__"
    text = (
        template.replace("{{inconsistency_items}}", inconsistency_token)
        .replace("{{developer_description}}", dev_token)
        .replace("{{code_derived_description}}", code_token)
    )
    return (
        text.replace(inconsistency_token, inconsistency_items)
        .replace(dev_token, developer_description)
        .replace(code_token, code_derived_description)
    )


def _build_chain(llm):
    prompt = ChatPromptTemplate.from_messages([("human", "{prompt_text}")])
    return prompt, llm


def _coerce_message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge output is not a JSON object")
    return value


def _extract_inconsistency_items(text: str) -> list[dict]:
    """Return the consistency report's inconsistency list."""
    payload = json.loads(text)
    items = payload.get("inconsistency", payload.get("inconsistencies", []))
    if not isinstance(items, list):
        items = []
    return [item for item in items if isinstance(item, dict)]


def _normalize_threats(raw_text: str, items: list[dict]) -> dict:
    """Keep the judge's threats that carry a valid security label."""
    item_count = len(items)
    payload = _extract_json_object(raw_text)
    raw_threats = payload.get("threats")
    if raw_threats is None and payload.get("security_label"):
        raw_threats = [payload]
    if not isinstance(raw_threats, list):
        raw_threats = []

    threats: list[dict] = []
    for raw in raw_threats:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("security_label", "")).strip().lower()
        if label == "clean" or label not in VALID_SECURITY_LABELS:
            continue
        try:
            index = int(raw.get("inconsistency_index", 0))
        except (TypeError, ValueError):
            index = 0
        if item_count > 0:
            index = min(max(index, 0), item_count - 1)
        threats.append(
            {
                "inconsistency_index": index,
                "security_label": label,
                "reason": str(raw.get("reason") or "").strip(),
            }
        )
    return {"threats": threats}


def _format_label(label: str) -> str:
    if label == "clean":
        return f"\033[32m{label}\033[0m"
    if label in {"potentially_malicious_code", "unsafe_implementation", "instruction_injection"}:
        return f"\033[31m{label}\033[0m"
    return f"\033[33m{label}\033[0m"


def _diag_log_path(log_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_root / f"judge_diagnostics_{ts}.jsonl"


def _append_diag(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _invoke(prompt_text: str, llm, prompt) -> tuple[str, dict]:
    messages = prompt.format_messages(prompt_text=prompt_text)
    response = llm.invoke(messages)
    text = _coerce_message_text(getattr(response, "content", ""))
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    diagnostics = {
        "response_metadata": response_metadata,
        "usage_metadata": usage_metadata,
        "finish_reason": response_metadata.get("finish_reason"),
    }
    return text, diagnostics


def create_judge_runtime(llm, log_root: Path) -> dict:
    log_root.mkdir(parents=True, exist_ok=True)
    diag_path = _diag_log_path(log_root)
    logger.info("Judge diagnostics will be written to %s", diag_path)
    threat_template = _read_prompt(PROMPT_THREAT_PATH)
    prompt, llm = _build_chain(llm)
    return {
        "prompt": prompt,
        "llm": llm,
        "diag_path": diag_path,
        "threat_template": threat_template,
    }


def process_judge_tool(
    server_name: str,
    tool: dict,
    runtime: dict,
    overwrite: bool = False,
    **kwargs,
) -> dict:
    developer_description = (tool.get("tool_description") or "").strip()
    description_path = tool.get("description_path")
    consistency_path = tool.get("consistency_check_path")

    if not developer_description or not description_path or not consistency_path:
        logger.info("  judge: skipped (missing description/consistency inputs)")
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    description_file = Path(description_path)
    consistency_file = Path(consistency_path)
    if not description_file.exists() or not consistency_file.exists():
        logger.info("  judge: skipped (description/consistency file not found)")
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    judge_file = description_file.with_name("judge")
    if judge_file.exists() and not overwrite:
        logger.info("  judge: skipped (exists)")
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    started_at = time.perf_counter()
    diagnostics: dict = {}
    try:
        code_derived_description = description_file.read_text(encoding="utf-8")
        consistency_report_text = consistency_file.read_text(encoding="utf-8")
        inconsistency_items = _extract_inconsistency_items(consistency_report_text)
        if not inconsistency_items:
            judge_file.unlink(missing_ok=True)
            for field in ("judge_path", "judge_value", "security_labels"):
                tool.pop(field, None)
            logger.info("  judge: skipped (empty inconsistency list)")
            return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

        inconsistency_json = json.dumps(inconsistency_items, ensure_ascii=False, indent=2)
        prompt_text = _render_prompt(
            runtime["threat_template"],
            inconsistency_items=inconsistency_json,
            developer_description=developer_description,
            code_derived_description=code_derived_description,
        )
        raw_text, diagnostics = _invoke(prompt_text, llm=runtime["llm"], prompt=runtime["prompt"])
        judgment = _normalize_threats(raw_text, inconsistency_items)
        labels = sorted({t["security_label"] for t in judgment["threats"]})

        judge_file.write_text(json.dumps(judgment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tool["judge_path"] = str(judge_file)
        tool["judge_value"] = judgment
        tool["security_labels"] = labels
        if labels:
            logger.info("  judge: labels=%s", ", ".join(_format_label(label) for label in labels))
        else:
            logger.info("  judge: %s", _format_label("clean"))
        logger.info("  judge: %s", judge_file)

        _append_diag(
            runtime["diag_path"],
            {
                "server_name": server_name,
                "tool_name": tool.get("tool_name", ""),
                "path": str(judge_file),
                "token_usage": diagnostics.get("usage_metadata", {}),
                "latency_sec": round(time.perf_counter() - started_at, 3),
            },
        )
        return {"total": 1, "generated": 1, "skipped": 0, "failed": 0}
    except Exception as exc:
        tool["judge_path"] = None
        tool["judge_value"] = None
        tool["security_labels"] = None
        _append_diag(
            runtime["diag_path"],
            {
                "server_name": server_name,
                "tool_name": tool.get("tool_name", ""),
                "path": str(judge_file),
                "token_usage": diagnostics.get("usage_metadata", {}),
                "latency_sec": round(time.perf_counter() - started_at, 3),
                "error": str(exc),
            },
        )
        logger.warning("  judge: FAILED (%s)", exc)
        return {"total": 1, "generated": 0, "skipped": 0, "failed": 1}
