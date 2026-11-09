import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tool_extract import extract_tools
from code_slicer import generate_slices
from desc_gen import create_description_runtime, process_description_tool
from judge import create_judge_runtime, process_judge_tool
from consistency_check import create_consistency_check_runtime, process_consistency_check_tool


PROJECT_ROOT = Path(__file__).parent
DEFAULT_LOG_ROOT = PROJECT_ROOT / "mcp_log"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "mcp_result"
LLM_CONFIG_PATH = PROJECT_ROOT / "llm_config.json"


def load_llm_configs(path: Path = LLM_CONFIG_PATH) -> dict[str, dict]:
    """Load and minimally validate named LLM configurations from JSON."""
    try:
        configs = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"LLM config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in LLM config file {path}: {exc}") from exc

    if not isinstance(configs, dict) or not configs:
        raise RuntimeError(f"LLM config file must contain a non-empty object: {path}")
    required = {"model", "base_url", "api_key_env"}
    for entry_name, config in configs.items():
        if not isinstance(config, dict):
            raise RuntimeError(f"LLM config entry {entry_name!r} must be an object")
        missing = sorted(required - config.keys())
        if missing:
            raise RuntimeError(
                f"LLM config entry {entry_name!r} is missing: {', '.join(missing)}"
            )
    return configs


def build_llm(entry_name: str, configs: dict[str, dict]) -> ChatOpenAI:
    """Build the shared LLM client from one JSON config entry."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    config = configs[entry_name]
    api_key_env = str(config["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"LLM config {entry_name!r} requires environment variable {api_key_env}"
        )

    llm_kwargs = {
        "model": str(config["model"]),
        "base_url": str(config["base_url"]),
        "api_key": api_key,
    }
    if "temperature" in config:
        llm_kwargs["temperature"] = float(config["temperature"])
    if "reasoning_effort" in config:
        llm_kwargs["reasoning_effort"] = str(config["reasoning_effort"])
    if "extra_body" in config:
        llm_kwargs["extra_body"] = config["extra_body"]
    return ChatOpenAI(**llm_kwargs)


def setup_logging() -> None:
    """Log concise pipeline progress to stdout."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def _save_tool_metadata(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_STAT_KEYS = ("generated", "skipped", "failed")
# Wall-clock limit (seconds) for one tool in each LLM stage.
_STAGE_TIMEOUT_SEC = {"desc-gen": 200, "consistency-check": 400, "judge": 200}


class _StageTimeoutError(BaseException):
    pass


class _StageTimeout:
    def __init__(self, timeout_sec: int):
        self.timeout_sec = int(timeout_sec)
        self._prev_handler = None

    def _handler(self, signum, frame):
        raise _StageTimeoutError(f"stage timed out after {self.timeout_sec}s")

    def __enter__(self):
        if self.timeout_sec <= 0:
            return self
        self._prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, float(self.timeout_sec))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timeout_sec <= 0:
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self._prev_handler is not None:
            signal.signal(signal.SIGALRM, self._prev_handler)


def _parse_single_tool_target(raw: str) -> tuple[str, str]:
    value = (raw or "").strip()
    if not value or "/" not in value:
        raise ValueError("--single-tool must be in format <server_name>/<tool_name>")
    server_name, tool_name = value.rsplit("/", 1)
    server_name = server_name.strip()
    tool_name = tool_name.strip()
    if not server_name or not tool_name:
        raise ValueError("--single-tool must be in format <server_name>/<tool_name>")
    return server_name, tool_name


def main():
    llm_configs = load_llm_configs()

    parser = argparse.ArgumentParser(description="MCP server checker")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Run the complete extract, slice, description, consistency-check, and judge pipeline")
    parser.add_argument("--extract", action="store_true", help="Extract tool metadata from locally downloaded server repositories")
    parser.add_argument("--slice", action="store_true", help="Generate code slices from tool_metadata.json")
    parser.add_argument("--source", type=Path, required=True, help="Directory whose immediate subdirectories are MCP server source trees (required)")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_ROOT, help="Directory for diagnostic logs (default: mcp_log/)")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_ROOT, help="Directory for slices and analysis results (default: mcp_result/)")
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        choices=sorted(llm_configs),
        help="LLM config entry from llm_config.json (required)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results for all selected stages")
    parser.add_argument("--desc-gen", action="store_true", help="Generate descriptions from generated code slices (agent mode with codebase search)")
    parser.add_argument("--consistency-check", action="store_true", help="Match original descriptions against code-derived operations and record inconsistencies")
    parser.add_argument("--judge", action="store_true", help="Classify consistency-check reports into threat labels")
    parser.add_argument("--single-tool", type=str, default="", help="Run slice and/or selected post-slice stages for one tool only: <server_name>/<tool_name>")
    args = parser.parse_args()

    if args.run_all:
        args.extract = True
        args.slice = True
        args.desc_gen = True
        args.consistency_check = True
        args.judge = True

    source_root = args.source.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"--source is not a directory: {source_root}")

    log_root = args.log.expanduser().resolve()
    result_root = args.result.expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log root: %s", log_root)
    logger.info("Result root: %s", result_root)
    logger.info("LLM config: %s", args.llm)

    tool_metadata_path = result_root / "tool_metadata.json"
    single_server_name = ""
    single_tool_name = ""
    if args.single_tool:
        try:
            single_server_name, single_tool_name = _parse_single_tool_target(args.single_tool)
        except ValueError as exc:
            parser.error(str(exc))
        logger.info("Single-tool mode enabled for server/tool (%r, %r)", single_server_name, single_tool_name)

    if args.extract:
        logger.info("Extracting tools from MCP server directories under %s ...", source_root)
        extract_results = extract_tools(source_root, output_path=tool_metadata_path)
        tool_count = sum(len(r.get("tools", [])) for r in extract_results)
        logger.info("Result: %d matched servers, %d tools → %s", len(extract_results), tool_count, tool_metadata_path)

    if args.slice:
        logger.info("Generating code slices from %s ...", tool_metadata_path)
        stats = generate_slices(
            tool_metadata_path,
            output_base=result_root,
            single_server_name=single_server_name,
            single_tool_name=single_tool_name,
            overwrite=args.overwrite,
        )
        logger.info("Result: %d/%d sliced (%d truncated), %d skipped, %d failed, %d timed out → %s",
                     stats["sliced"], stats["total"], stats["truncated"],
                     stats["skipped"], stats["failed"], stats["timed_out"], result_root)
        if args.single_tool and stats.get("total", 0) == 0:
            logger.warning("No matching target sliced for --single-tool=%s (check names)", args.single_tool)

    llm_steps: list[tuple[str, object, dict]] = []
    if args.desc_gen or args.consistency_check or args.judge:
        llm = build_llm(args.llm, llm_configs)
    if args.desc_gen:
        # A full re-run must not inherit memory from earlier runs; a single-tool
        # re-run keeps the memory its server accumulated.
        runtime = create_description_runtime(
            llm=llm, log_root=log_root, reset_memory=args.overwrite and not args.single_tool,
        )
        llm_steps.append(("desc-gen", process_description_tool, runtime))
    if args.consistency_check:
        runtime = create_consistency_check_runtime(llm=llm, log_root=log_root)
        llm_steps.append(("consistency-check", process_consistency_check_tool, runtime))
    if args.judge:
        runtime = create_judge_runtime(llm=llm, log_root=log_root)
        llm_steps.append(("judge", process_judge_tool, runtime))

    if llm_steps:
        logger.info("Running post-slice pipeline from %s ...", tool_metadata_path)
        data = json.loads(tool_metadata_path.read_text(encoding="utf-8"))
        server_records = data.get("results", [])
        stage_stats = {name: dict.fromkeys(_STAT_KEYS, 0) for name, _, _ in llm_steps}

        tool_idx = 0
        matched = 0
        for server in server_records:
            server_name = server.get("name", "unknown")
            if single_server_name and server_name != single_server_name:
                continue
            repo_dir = server.get("local_repo_dir", "")
            for tool in server.get("tools", []):
                tool_name = tool.get("tool_name", "")
                if single_tool_name and tool_name != single_tool_name:
                    continue
                matched += 1
                slice_timed_out = tool.get("timeout_stage") == "slice"
                if tool.get("timeout") is True and (slice_timed_out or not args.overwrite):
                    # A slicing timeout leaves no slice, so later stages are skipped even
                    # with --overwrite; re-run --slice --overwrite to retry slicing.
                    logger.info(
                        "[-] tool: \033[90m%s/%s\033[0m (skipped: timeout=true, stage=%s)",
                        server_name, tool_name, tool.get("timeout_stage"),
                    )
                    continue
                if args.overwrite:
                    tool.pop("timeout", None)
                    tool.pop("timeout_stage", None)
                tool_idx += 1
                logger.info("[%d] tool: \033[34m%s/%s\033[0m", tool_idx, server_name, tool_name)
                for name, processor, runtime in llm_steps:
                    timeout_sec = _STAGE_TIMEOUT_SEC[name]
                    started_at = time.perf_counter()
                    try:
                        with _StageTimeout(timeout_sec):
                            stats = processor(
                                server_name, tool, runtime, overwrite=args.overwrite, local_repo_dir=repo_dir,
                            )
                        logger.info("  %s elapsed: %.1fs", name, time.perf_counter() - started_at)
                        for key in _STAT_KEYS:
                            stage_stats[name][key] += stats.get(key, 0)
                        _save_tool_metadata(tool_metadata_path, data)
                    except _StageTimeoutError:
                        tool["timeout"] = True
                        tool["timeout_stage"] = name
                        logger.warning("  %s: timeout after %ds, skip remaining stages for this tool", name, timeout_sec)
                        stage_stats[name]["failed"] += 1
                        _save_tool_metadata(tool_metadata_path, data)
                        break
                logger.info("")

        if args.single_tool and matched == 0:
            logger.warning("No matching tool processed for --single-tool=%s (check server and tool names)", args.single_tool)

        for name, _, runtime in llm_steps:
            stats = stage_stats[name]
            logger.info(
                "%s result: %d generated, %d skipped, %d failed ; diagnostics=%s",
                name, stats["generated"], stats["skipped"], stats["failed"], runtime["diag_path"],
            )


if __name__ == "__main__":
    main()
