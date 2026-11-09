"""Generate tool descriptions from code slices via an agent with codebase search tools."""

import hashlib
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool

from code_slicer import SUPPORTED_EXTENSIONS, _BaseSlicer, _make_slicer

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompt" / "desc_gen.md"
MAX_TOOL_CALLS = 10
_READ_FILE_MAX_LINES = 200
_GREP_MAX_RESULTS = 60
_LIST_DIR_MAX_RESULTS = 200
_FIND_DEF_MAX_FILES = 40
_FIND_DEF_MAX_MATCHES = 5
_FIND_DEF_MAX_LINES = 150
# Vendored, built and minified code is not the server's own source.
_FIND_DEF_EXCLUDE_GLOBS = (
    "!**/node_modules/**", "!**/dist/**", "!**/build/**", "!**/.venv/**",
    "!**/venv/**", "!**/site-packages/**", "!*.d.ts", "!*.min.js",
)
# When the agent sets file_glob to code-only (e.g. "*.py"), ripgrep would skip README / docs where
# env vars and deployment notes are explained, which weakens code-derived description and consistency checks.
_GREP_DOC_GLOBS = ("*.md", "*.markdown", "*.mdx", "*.rst", "README*")
_SERVER_MEMORY_FILENAME = "desc_gen_memory.md"
_SERVER_MEMORY_MAX_CHARS = 6000
_SERVER_MEMORY_ITEM_MAX_CHARS = 240

# Required <description> structure; consistency-check relies on these sections.
_REQUIRED_SECTIONS = (
    ("**1. Primary Function**", re.compile(r"^[ \t]*\**[ \t]*1\.[ \t]*Primary Function\b", re.M | re.I)),
    (
        "**2. Inputs and Runtime Dependencies**",
        re.compile(r"^[ \t]*\**[ \t]*2\.[ \t]*Inputs and Runtime Dependencies\b", re.M | re.I),
    ),
    ("**3. Operations**", re.compile(r"^[ \t]*\**[ \t]*3\.[ \t]*Operations\b", re.M | re.I)),
)
_STEP_RE = re.compile(r"^[ \t]*-[ \t]*Step[ \t]*:", re.M | re.I)
_CRITICAL_RE = re.compile(r"^[ \t]*-?[ \t]*Critical[ \t]*:[ \t]*(yes|no)\b", re.M | re.I)


# ---------------------------------------------------------------------------
# Agent tools — created per-repo via closure
# ---------------------------------------------------------------------------

def _make_tools(repo_dir: Path) -> list:
    """Build agent tools scoped to *repo_dir*."""
    repo_root = repo_dir.resolve()

    @langchain_tool
    def grep_repo(pattern: str, file_glob: str = "") -> str:
        """Search the repository for a regex pattern using ripgrep.

        Args:
            pattern: The regex pattern to search for.
            file_glob: Optional glob to filter files (e.g. "*.py", "*.ts"). When set, Markdown and
                README* paths are searched as well so credential/env documentation is not missed.

        Returns matching lines with relative file paths and line numbers.
        """
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", "30",
               "--max-columns", "300", "--color", "never"]
        if file_glob:
            cmd += ["--glob", file_glob]
            for doc_glob in _GREP_DOC_GLOBS:
                cmd += ["--glob", doc_glob]
        cmd += ["--", pattern, "."]
        try:
            result = subprocess.run(
                cmd, cwd=str(repo_dir), capture_output=True, text=True,
                timeout=15,
            )
            output = result.stdout.strip()
            if not output:
                return "(no matches)"
            lines = output.splitlines()
            if len(lines) > _GREP_MAX_RESULTS:
                lines = lines[:_GREP_MAX_RESULTS]
                lines.append(f"... (truncated, showing first {_GREP_MAX_RESULTS} matches)")
            return "\n".join(lines)
        except FileNotFoundError:
            return "(ripgrep 'rg' not found on this system)"
        except subprocess.TimeoutExpired:
            return "(grep timed out after 15s)"

    @langchain_tool
    def read_file(relative_path: str, start_line: int = 1, end_line: int = 50) -> str:
        """Read a section of a file from the repository.

        Args:
            relative_path: Path relative to the repository root.
            start_line: First line to read (1-indexed, default 1).
            end_line: Last line to read (inclusive, default 50). Capped at start_line + 199.
        """
        target = repo_dir / relative_path
        if not target.exists():
            return f"(file not found: {relative_path})"
        try:
            resolved = target.resolve()
            resolved.relative_to(repo_root)
        except ValueError:
            return "(path escapes repository root)"
        except (OSError, RuntimeError):
            return "(invalid path)"
        try:
            all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"(read error: {exc})"
        total = len(all_lines)
        start = max(1, start_line)
        end = min(total, max(start, end_line))
        end = min(end, start + _READ_FILE_MAX_LINES - 1)
        selected = all_lines[start - 1 : end]
        numbered = [f"{start + i:>6}|{line}" for i, line in enumerate(selected)]
        header = f"[{relative_path}] lines {start}-{end} of {total}"
        return header + "\n" + "\n".join(numbered)

    @langchain_tool
    def list_dir(relative_path: str = ".") -> str:
        """List files and directories at a path in the repository.

        Args:
            relative_path: Path relative to the repository root (default ".").
        """
        target = repo_dir / relative_path
        if not target.exists():
            return f"(directory not found: {relative_path})"
        if not target.is_dir():
            return f"({relative_path} is a file, not a directory)"
        try:
            resolved = target.resolve()
            resolved.relative_to(repo_root)
        except ValueError:
            return "(path escapes repository root)"
        except (OSError, RuntimeError):
            return "(invalid path)"
        entries: list[str] = []
        try:
            for child in sorted(target.iterdir()):
                suffix = "/" if child.is_dir() else ""
                rel = child.relative_to(repo_dir)
                entries.append(f"  {rel}{suffix}")
        except Exception as exc:
            return f"(list error: {exc})"
        if not entries:
            return "(empty directory)"
        entry_count = len(entries)
        if len(entries) > _LIST_DIR_MAX_RESULTS:
            entries = entries[:_LIST_DIR_MAX_RESULTS]
            entries.append(
                f"... (truncated, showing first {_LIST_DIR_MAX_RESULTS} entries)"
            )
        return f"[{relative_path}] {entry_count} entries:\n" + "\n".join(entries)

    slicers: dict[str, _BaseSlicer] = {}

    def _slicer_for(path: Path) -> _BaseSlicer:
        key = "py" if path.suffix.lower() == ".py" else "js"
        if key not in slicers:
            slicers[key] = _make_slicer(str(repo_root), path.suffix.lower())
        return slicers[key]

    def _candidate_files(token: str) -> list[Path]:
        cmd = ["rg", "--files-with-matches", "--fixed-strings", "--word-regexp", "--color", "never"]
        for ext in sorted(SUPPORTED_EXTENSIONS):
            cmd += ["--glob", f"*{ext}"]
        for glob in _FIND_DEF_EXCLUDE_GLOBS:
            cmd += ["--glob", glob]
        cmd += ["--", token, "."]
        result = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=15)
        return [repo_root / line for line in sorted(result.stdout.splitlines())[:_FIND_DEF_MAX_FILES]]

    @langchain_tool
    def find_definition(symbol: str) -> str:
        """Return the source of a function, method, class, or module-level variable.

        Use this first when you need to know what a symbol does; it saves a
        grep_repo + read_file round trip. Comments and docstrings are removed,
        as in the code slice.

        Args:
            symbol: A name such as "send_request" or "Client.send", or a
                "path::name" entry from a truncated slice's omitted-function list.
        """
        path_part, _, name = symbol.strip().rpartition("::")
        name = name.strip()
        if not name:
            return "(empty symbol)"
        if path_part:
            target = repo_dir / path_part.strip()
            try:
                target.resolve().relative_to(repo_root)
            except ValueError:
                return "(path escapes repository root)"
            except (OSError, RuntimeError):
                return "(invalid path)"
            if not target.is_file():
                return f"(file not found: {path_part.strip()})"
            if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return f"(unsupported file type: {path_part.strip()}; use read_file)"
            files = [target.resolve()]
        else:
            try:
                files = _candidate_files(name.rsplit(".", 1)[-1])
            except FileNotFoundError:
                return "(ripgrep 'rg' not found on this system)"
            except subprocess.TimeoutExpired:
                return "(definition search timed out after 15s)"

        blocks: list[str] = []
        for path in files:
            filepath = str(path.resolve())
            slicer = _slicer_for(path)
            try:
                definitions = slicer.find_definitions(filepath, name)
            except Exception as exc:  # unreadable or unparsable file
                logger.debug("find_definition skipped %s: %s", path, exc)
                continue
            rel = path.relative_to(repo_root).as_posix()
            for kind, qualified, node in definitions:
                start, end = node.start_point[0] + 1, node.end_point[0] + 1
                lines = slicer._source_text(filepath, node.start_byte, node.end_byte).splitlines()
                if len(lines) > _FIND_DEF_MAX_LINES:
                    lines = lines[:_FIND_DEF_MAX_LINES] + [
                        f"... (truncated; read_file {rel} lines {start}-{end} for the rest)"
                    ]
                blocks.append(f"[{rel}:{start}-{end}] {kind} {qualified}\n" + "\n".join(lines))
                if len(blocks) >= _FIND_DEF_MAX_MATCHES:
                    blocks.append(f"... (stopped at {_FIND_DEF_MAX_MATCHES} definitions; qualify the name to narrow)")
                    return "\n\n".join(blocks)
        if not blocks:
            return f"(no definition of '{name}' found; try grep_repo)"
        return "\n\n".join(blocks)

    return [find_definition, grep_repo, read_file, list_dir]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_prompt(template: str, tool_name: str, code_slice: str, server_memory: str) -> str:
    values = {
        "TOOL_NAME": tool_name,
        "CODE_SLICE": code_slice,
        "SERVER_MEMORY": server_memory,
    }
    return re.sub(
        r"\{(TOOL_NAME|CODE_SLICE|SERVER_MEMORY)\}",
        lambda match: values[match.group(1)],
        template,
    )


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


def _usage_value(usage: dict, *keys: str) -> int:
    value = usage
    for key in keys:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _merge_usage(total: dict, usage: dict) -> dict:
    """Accumulate common token usage fields across agent LLM calls."""
    if not isinstance(usage, dict):
        return total
    total["input_tokens"] = total.get("input_tokens", 0) + _usage_value(usage, "input_tokens")
    total["output_tokens"] = total.get("output_tokens", 0) + _usage_value(usage, "output_tokens")
    total["total_tokens"] = total.get("total_tokens", 0) + _usage_value(usage, "total_tokens")
    total.setdefault("input_token_details", {})
    total.setdefault("output_token_details", {})
    total["input_token_details"]["cache_read"] = total["input_token_details"].get("cache_read", 0) + _usage_value(
        usage, "input_token_details", "cache_read"
    )
    total["output_token_details"]["reasoning"] = total["output_token_details"].get("reasoning", 0) + _usage_value(
        usage, "output_token_details", "reasoning"
    )
    return total


def _looks_truncated(desc: str, finish_reason: Optional[str]) -> bool:
    return bool(desc.strip()) and str(finish_reason or "").lower() == "length"


def _looks_like_tool_call_text(desc: str) -> bool:
    text = (desc or "").strip()
    if not text:
        return False
    markers = ("<tool_call>", "</tool_call>", "<arg_key>", "<arg_value>")
    return any(marker in text for marker in markers)


def _format_errors(desc: str) -> list[str]:
    """Return the ways *desc* deviates from the required <description> format."""
    if not desc.strip():
        return ["empty output"]
    errors: list[str] = []
    if _looks_like_tool_call_text(desc):
        errors.append("contains raw tool-call markup instead of a description")
    for label, pattern in _REQUIRED_SECTIONS:
        if not pattern.search(desc):
            errors.append(f"missing section {label}")
    operations = _REQUIRED_SECTIONS[2][1].search(desc)
    if operations:
        body = desc[operations.end():]
        starts = [match.start() for match in _STEP_RE.finditer(body)]
        if not starts:
            errors.append("the Operations section has no '- Step:' entries")
        else:
            bounds = starts[1:] + [len(body)]
            missing = sum(
                1 for start, end in zip(starts, bounds)
                if not _CRITICAL_RE.search(body, start, end)
            )
            if missing:
                errors.append(
                    f"{missing} of {len(starts)} operation steps lack a 'Critical: yes' or 'Critical: no' line"
                )
    return errors


def _server_memory_path_from_slice(slice_file: Path) -> Path:
    return slice_file.parent.parent / _SERVER_MEMORY_FILENAME


def _reset_server_memory_once(memory_path: Path, runtime: dict) -> None:
    """Delete a server's memory file the first time this run touches that server.

    A full ``--overwrite`` regenerates every description, so memory left by an
    earlier run (another model, prompt or repository revision) must not be fed
    back in as "confirmed" definitions.
    """
    done = runtime.setdefault("memory_reset_paths", set())
    if memory_path in done:
        return
    done.add(memory_path)
    if memory_path.exists():
        memory_path.unlink()
        logger.info("  desc-gen: cleared server memory from a previous run (%s)", memory_path)


def _read_server_memory(memory_path: Path) -> str:
    if not memory_path.exists():
        return "(no server memory yet)"
    text = memory_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return "(no server memory yet)"
    return text[:_SERVER_MEMORY_MAX_CHARS]


def _extract_tag_block(text: str, tag_name: str) -> str:
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def _extract_loose_tag_block(text: str, tag_name: str) -> str:
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        next_tag = text.find("<", start)
        end = next_tag if next_tag != -1 else len(text)
    return text[start:end].strip()


def _strip_loose_description_tags(text: str) -> str:
    text = (text or "").strip()
    if "<memory_update>" in text:
        text = text.split("<memory_update>", 1)[0].strip()
    if text.startswith("<description>"):
        text = text[len("<description>"):].strip()
    if text.endswith("</description>"):
        text = text[: -len("</description>")].strip()
    return text


def _split_description_and_memory_update(text: str) -> tuple[str, list[str]]:
    description = _extract_tag_block(text, "description")
    memory_block = _extract_tag_block(text, "memory_update") or _extract_loose_tag_block(text, "memory_update")
    if not description:
        description = _strip_loose_description_tags(text)

    items: list[str] = []
    if memory_block and memory_block.upper() != "NONE":
        for raw_line in memory_block.splitlines():
            line = raw_line.strip()
            if not line or line.upper() == "NONE":
                continue
            # Strip optional leading "- " from bullet-list output
            if line.startswith("-"):
                line = line[1:].strip()
            if not line:
                continue
            # Enforce symbol: description format — skip lines without ":"
            if ":" not in line:
                continue
            line = " ".join(line.split())
            if len(line) > _SERVER_MEMORY_ITEM_MAX_CHARS:
                line = line[: _SERVER_MEMORY_ITEM_MAX_CHARS].rstrip()
            items.append(line)
    return description, items


def _symbol_key(entry: str) -> str:
    """Return the symbol name (part before first ':') for deduplication."""
    return entry.split(":", 1)[0].strip().lower()


def _append_server_memory(memory_path: Path, items: list[str]) -> None:
    if not items:
        return
    existing_lines: list[str] = []
    existing_symbols: set[str] = set()
    if memory_path.exists():
        for line in memory_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            existing_lines.append(stripped)
            if ":" in stripped:
                existing_symbols.add(_symbol_key(stripped))

    new_items: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = _symbol_key(normalized)
        if key in existing_symbols:
            continue
        existing_symbols.add(key)
        new_items.append(normalized)

    if not new_items:
        return

    existing_lines.extend(new_items)
    while len("\n".join(existing_lines)) > _SERVER_MEMORY_MAX_CHARS:
        existing_lines.pop(0)
    memory_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")


def _diag_log_path(log_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_root / f"desc_gen_diagnostics_{ts}.jsonl"


def _append_diag(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _execute_tool_call(tool_call: dict, tool_map: dict) -> str:
    """Dispatch a single tool call and return the string result."""
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    fn = tool_map.get(name)
    if fn is None:
        return f"(unknown tool: {name})"
    try:
        return fn.invoke(args)
    except Exception as exc:
        return f"(tool error: {exc})"


def _invoke_description_agent(
    prompt_text: str,
    llm,
    tools: Optional[list],
) -> tuple[str, dict, list[BaseMessage]]:
    """Run the agent loop: LLM may call tools, then produces the final description.

    Without *tools* this is a single LLM call. Returns the final text, diagnostics
    (usage, finish reason, rounds and a trace of tool names/arguments, never tool
    output contents) and the conversation, so a format retry can continue it.
    """
    messages: list = [HumanMessage(content=prompt_text)]
    total_usage: dict = {}
    trace: list[dict] = []
    iterations = 0
    tool_call_count = 0
    final_response = None
    hit_tool_call_limit = False

    if not tools:
        final_response = llm.invoke(messages)
        _merge_usage(total_usage, getattr(final_response, "usage_metadata", {}) or {})
        messages.append(final_response)
    else:
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {item.name: item for item in tools}
        while tool_call_count < MAX_TOOL_CALLS:
            iterations += 1
            response = llm_with_tools.invoke(messages)
            _merge_usage(total_usage, getattr(response, "usage_metadata", {}) or {})
            messages.append(response)

            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                final_response = response
                break

            for tc in calls:
                entry = {"round": iterations, "name": tc.get("name", ""), "args": tc.get("args", {})}
                if tool_call_count < MAX_TOOL_CALLS:
                    result_text = _execute_tool_call(tc, tool_map)
                    tool_call_count += 1
                    entry["result_chars"] = len(result_text)
                else:
                    result_text = "(tool-call limit reached; call not executed)"
                    entry["executed"] = False
                trace.append(entry)
                messages.append(
                    ToolMessage(content=result_text, tool_call_id=tc["id"])
                )

            if tool_call_count >= MAX_TOOL_CALLS:
                hit_tool_call_limit = True
                break

        if hit_tool_call_limit:
            messages.append(
                HumanMessage(
                    content=(
                        "You have reached the maximum number of tool calls. "
                        "Do not call any more tools. Based only on the code slice and the "
                        "tool outputs already shown in this conversation, write the final "
                        "description now using the required format. If some behavior "
                        "is still uncertain, describe only what the available evidence shows."
                    )
                )
            )
            final_response = llm.invoke(messages)
            _merge_usage(total_usage, getattr(final_response, "usage_metadata", {}) or {})
            messages.append(final_response)

    text = _coerce_message_text(getattr(final_response, "content", ""))
    response_metadata = getattr(final_response, "response_metadata", {}) or {}
    diagnostics = {
        "usage_metadata": total_usage,
        "finish_reason": response_metadata.get("finish_reason"),
        "agent_rounds": iterations,
        "tool_call_count": tool_call_count,
        "tool_calls": trace,
    }
    return text, diagnostics, messages


def _retry_with_format_feedback(
    messages: list[BaseMessage],
    llm,
    errors: list[str],
    diagnostics: dict,
) -> str:
    """Ask the model once, in the same conversation, to fix a malformed answer."""
    messages.append(
        HumanMessage(
            content=(
                "Your previous answer does not follow the required output format: "
                + "; ".join(errors)
                + ". Do not call any tools. Rewrite the complete answer now in exactly the "
                "required format: a <description> block containing **1. Primary Function**, "
                "**2. Inputs and Runtime Dependencies** and **3. Operations**, where every "
                "'- Step:' is followed by a 'Critical: yes' or 'Critical: no' line and a "
                "'Supporting:' line, then a "
                "<memory_update> block. Keep the same findings; only fix the format."
            )
        )
    )
    response = llm.invoke(messages)
    messages.append(response)
    _merge_usage(diagnostics["usage_metadata"], getattr(response, "usage_metadata", {}) or {})
    diagnostics["finish_reason"] = (getattr(response, "response_metadata", {}) or {}).get("finish_reason")
    return _coerce_message_text(getattr(response, "content", ""))


# ---------------------------------------------------------------------------
# Runtime & processing
# ---------------------------------------------------------------------------

def _stage_result(*, generated: int = 0, skipped: int = 0, failed: int = 0) -> dict:
    return {"total": 1, "generated": generated, "skipped": skipped, "failed": failed}


def _set_description_result(tool: dict, path: Optional[Path]) -> None:
    tool["description_generated"] = path is not None
    tool["description_path"] = str(path) if path is not None else None


def create_description_runtime(llm, log_root: Path, reset_memory: bool = False) -> dict:
    """Build the description-stage runtime.

    *reset_memory* deletes each server's memory file before the first tool of
    that server is regenerated with ``overwrite``; use it for full re-runs, not
    single-tool runs.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    diag_path = _diag_log_path(log_root)
    logger.info("Description diagnostics will be written to %s", diag_path)
    return {
        "prompt_template": PROMPT_PATH.read_text(encoding="utf-8"),
        "llm": llm,
        "diag_path": diag_path,
        "reset_memory": reset_memory,
    }


def process_description_tool(
    server_name: str,
    tool: dict,
    runtime: dict,
    overwrite: bool = False,
    local_repo_dir: str = "",
) -> dict:
    slice_path = tool.get("slice_path")
    if not slice_path:
        _set_description_result(tool, None)
        return _stage_result(skipped=1)

    slice_file = Path(slice_path)
    if not slice_file.is_file():
        _set_description_result(tool, None)
        return _stage_result(skipped=1)

    description_file = slice_file.with_name("description")
    if (
        not overwrite
        and description_file.is_file()
        and description_file.stat().st_size > 0
    ):
        _set_description_result(tool, description_file)
        logger.info("  desc-gen: skipped (exists)")
        return _stage_result(skipped=1)

    started_at = time.perf_counter()
    diagnostics: dict = {}
    memory_info: dict = {}
    try:
        code_slice = slice_file.read_text(encoding="utf-8")
        memory_path = _server_memory_path_from_slice(slice_file)
        if overwrite and runtime.get("reset_memory"):
            _reset_server_memory_once(memory_path, runtime)
        server_memory = _read_server_memory(memory_path)
        # The memory snapshot makes a description reproducible: its content depends
        # on which tools of the same server ran earlier.
        memory_info = {
            "server_memory_sha256": hashlib.sha256(server_memory.encode("utf-8")).hexdigest(),
            "server_memory": server_memory,
        }
        prompt_text = _render_prompt(
            runtime["prompt_template"],
            tool_name=tool.get("tool_name", ""),
            code_slice=code_slice,
            server_memory=server_memory,
        )

        repo_path = Path(local_repo_dir) if local_repo_dir else None
        tools_list = _make_tools(repo_path) if repo_path and repo_path.is_dir() else None
        desc, diagnostics, messages = _invoke_description_agent(
            prompt_text,
            llm=runtime["llm"],
            tools=tools_list,
        )

        desc_body, memory_update_items = _split_description_and_memory_update(desc)
        suspected_truncation = _looks_truncated(desc_body, diagnostics.get("finish_reason"))
        format_errors = [] if suspected_truncation else _format_errors(desc_body)
        first_attempt_errors = None
        if format_errors:
            first_attempt_errors = format_errors
            logger.info("  desc-gen: retrying malformed output (%s)", "; ".join(format_errors))
            desc = _retry_with_format_feedback(messages, runtime["llm"], format_errors, diagnostics)
            desc_body, memory_update_items = _split_description_and_memory_update(desc)
            suspected_truncation = _looks_truncated(desc_body, diagnostics.get("finish_reason"))
            format_errors = [] if suspected_truncation else _format_errors(desc_body)

        desc_chars = len(desc_body)
        validation_error = None
        if suspected_truncation:
            validation_error = "truncated output"
        elif format_errors:
            validation_error = "; ".join(format_errors)

        diag_payload = {
            "server_name": server_name,
            "tool_name": tool.get("tool_name", ""),
            "path": str(description_file),
            "token_usage": diagnostics.get("usage_metadata", {}),
            "latency_sec": round(time.perf_counter() - started_at, 3),
            "finish_reason": diagnostics.get("finish_reason"),
            "agent_rounds": diagnostics.get("agent_rounds", 0),
            "tool_call_count": diagnostics.get("tool_call_count", 0),
            "tool_calls": diagnostics.get("tool_calls", []),
            "format_retry": first_attempt_errors is not None,
            **memory_info,
        }
        if first_attempt_errors:
            diag_payload["first_attempt_format_errors"] = first_attempt_errors
        if validation_error:
            diag_payload["error"] = validation_error
        _append_diag(runtime["diag_path"], diag_payload)

        if validation_error:
            _set_description_result(tool, None)
            logger.warning("  desc-gen: FAILED (%s, chars=%d)", validation_error, desc_chars)
            return _stage_result(failed=1)

        description_file.write_text(desc_body, encoding="utf-8")
        _append_server_memory(memory_path, memory_update_items)
        _set_description_result(tool, description_file)
        logger.info(
            "  desc-gen: %s (chars=%d, rounds=%d, tool_calls=%d%s)",
            description_file,
            desc_chars,
            diagnostics.get("agent_rounds", 0),
            diagnostics.get("tool_call_count", 0),
            ", format retry" if first_attempt_errors else "",
        )
        return _stage_result(generated=1)
    except Exception as exc:
        _set_description_result(tool, None)
        _append_diag(
            runtime["diag_path"],
            {
                "server_name": server_name,
                "tool_name": tool.get("tool_name", ""),
                "path": str(description_file),
                "token_usage": diagnostics.get("usage_metadata", {}),
                "latency_sec": round(time.perf_counter() - started_at, 3),
                "error": str(exc),
                **memory_info,
            },
        )
        logger.warning("  desc-gen: FAILED (%s)", exc)
        return _stage_result(failed=1)
