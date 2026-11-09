"""Check consistency between developer-provided and code-derived tool descriptions."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt" / "consistency_check.md"

VALID_STATUS = {"clean", "inconsistent"}
VALID_INCONSISTENCY_TYPES = {"code_extra", "description_extra", "conflict"}


def _read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _render_prompt(
    template: str,
    developer_description: str,
    code_derived_description: str,
) -> str:
    developer_description = f"{developer_description}\n"
    dev_token = f"__CDCHECKER_DEV_DESC_{uuid.uuid4().hex}__"
    code_token = f"__CDCHECKER_CODE_DESC_{uuid.uuid4().hex}__"
    text = (
        template.replace("{{developer_description}}", dev_token)
        .replace("{{code_derived_description}}", code_token)
    )
    return (
        text.replace(dev_token, developer_description)
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
        raise ValueError("consistency check output is not a JSON object")
    return value


def _norm_text(value, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text or default


def _normalize_inconsistencies(raw_items) -> list[dict]:
    if not isinstance(raw_items, list):
        return []
    items: list[dict] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        kind = _norm_text(raw.get("type"), "code_extra").lower()
        if kind not in VALID_INCONSISTENCY_TYPES:
            kind = "code_extra"
        items.append(
            {
                "type": kind,
                "code": _norm_text(raw.get("code", raw.get("generated_operation")), "NONE"),
                "description": _norm_text(raw.get("description", raw.get("original_description_part")), "NONE"),
                "reason": _norm_text(raw.get("reason"), ""),
            }
        )
    return items


def _normalize_report(raw_text: str) -> dict:
    payload = _extract_json_object(raw_text)
    inconsistency = _normalize_inconsistencies(
        payload.get("inconsistency", payload.get("inconsistencies"))
    )
    unverified_inconsistency = _normalize_inconsistencies(
        payload.get(
            "unverified_inconsistency",
            payload.get("unverified_inconsistencies"),
        )
    )
    status = _norm_text(payload.get("status"), "").lower()
    if status not in VALID_STATUS:
        status = "inconsistent" if inconsistency else "clean"
    if inconsistency and status == "clean":
        status = "inconsistent"
    if not inconsistency:
        status = "clean"
    return {
        "status": status,
        "inconsistency": inconsistency,
        "unverified_inconsistency": unverified_inconsistency,
    }


def _diag_log_path(log_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_root / f"consistency_check_diagnostics_{ts}.jsonl"


def _append_diag(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _invoke_consistency_check(prompt_text: str, llm, prompt) -> tuple[str, dict]:
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


def create_consistency_check_runtime(llm, log_root: Path) -> dict:
    log_root.mkdir(parents=True, exist_ok=True)
    diag_path = _diag_log_path(log_root)
    logger.info("Consistency-check diagnostics will be written to %s", diag_path)
    prompt_template = _read_prompt()
    prompt, llm = _build_chain(llm)
    return {
        "prompt_template": prompt_template,
        "prompt": prompt,
        "llm": llm,
        "diag_path": diag_path,
    }


def process_consistency_check_tool(
    server_name: str,
    tool: dict,
    runtime: dict,
    overwrite: bool = False,
    **kwargs,
) -> dict:
    developer_description = (tool.get("tool_description") or "").strip()
    description_path = tool.get("description_path")
    if not developer_description or not description_path:
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    description_file = Path(description_path)
    if not description_file.exists():
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    report_file = description_file.with_name("consistency_check")
    if tool.get("consistency_check_generated") and report_file.exists() and not overwrite:
        logger.info("  consistency-check: skipped (exists)")
        return {"total": 1, "generated": 0, "skipped": 1, "failed": 0}

    started_at = time.perf_counter()
    diagnostics: dict = {}
    try:
        code_derived_description = description_file.read_text(encoding="utf-8")
        prompt_text = _render_prompt(
            runtime["prompt_template"],
            developer_description=developer_description,
            code_derived_description=code_derived_description,
        )
        raw_text, diagnostics = _invoke_consistency_check(
            prompt_text,
            llm=runtime["llm"],
            prompt=runtime["prompt"],
        )
        report = _normalize_report(raw_text)
        critical_count = sum(
            1 for item in report["inconsistency"] if item.get("type") in {"code_extra", "conflict"}
        )
        _append_diag(
            runtime["diag_path"],
            {
                "server_name": server_name,
                "tool_name": tool.get("tool_name", ""),
                "path": str(report_file),
                "token_usage": diagnostics.get("usage_metadata", {}),
                "latency_sec": round(time.perf_counter() - started_at, 3),
            },
        )

        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tool["consistency_check_generated"] = True
        tool["consistency_check_path"] = str(report_file)
        tool["consistency_status"] = report["status"]
        tool["consistency_inconsistency_count"] = len(report["inconsistency"])

        # A new report invalidates any judge result derived from the previous one.
        description_file.with_name("judge").unlink(missing_ok=True)
        for field in ("judge_path", "judge_value", "security_labels"):
            tool.pop(field, None)
        logger.info("  consistency-check: %s", report_file)
        if report["status"] == "clean":
            logger.info("  consistency-check: \033[32mclean\033[0m")
        else:
            logger.info(
                "  consistency-check: \033[33m%s\033[0m (%d inconsistencies, %d critical)",
                report["status"],
                len(report["inconsistency"]),
                critical_count,
            )
        if report["unverified_inconsistency"]:
            logger.info(
                "  consistency-check: %d unverified external-service inconsistencies",
                len(report["unverified_inconsistency"]),
            )
        return {"total": 1, "generated": 1, "skipped": 0, "failed": 0}
    except Exception as exc:
        tool["consistency_check_generated"] = False
        tool["consistency_check_path"] = None
        tool["consistency_status"] = None
        tool["consistency_inconsistency_count"] = None
        _append_diag(
            runtime["diag_path"],
            {
                "server_name": server_name,
                "tool_name": tool.get("tool_name", ""),
                "path": str(report_file),
                "token_usage": diagnostics.get("usage_metadata", {}),
                "latency_sec": round(time.perf_counter() - started_at, 3),
                "error": str(exc),
            },
        )
        logger.warning("  consistency-check: FAILED (%s)", exc)
        return {"total": 1, "generated": 0, "skipped": 0, "failed": 1}
