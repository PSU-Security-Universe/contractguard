"""Extract MCP tool metadata from local source trees using tree-sitter."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

import tree_sitter_javascript as tsjs
import tree_sitter_python as tsp
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

MAX_SOURCE_BYTES = 2 * 1024 * 1024

PY_LANG = Language(tsp.language())
JS_LANG = Language(tsjs.language())
TS_LANG = Language(tsts.language_typescript())
TSX_LANG = Language(tsts.language_tsx())

PY_PARSER = Parser(PY_LANG)
JS_PARSER = Parser(JS_LANG)
TS_PARSER = Parser(TS_LANG)
TSX_PARSER = Parser(TSX_LANG)

LANG_MAP = {
    ".py": ("python", PY_PARSER),
    ".js": ("javascript", JS_PARSER),
    ".cjs": ("javascript", JS_PARSER),
    ".jsx": ("javascript", JS_PARSER),
    ".mjs": ("javascript", JS_PARSER),
    ".ts": ("typescript", TS_PARSER),
    ".cts": ("typescript", TS_PARSER),
    ".mts": ("typescript", TS_PARSER),
    ".tsx": ("typescript", TSX_PARSER),
}

MCP_SOURCE_MARKERS = (
    b"@modelcontextprotocol",
    b"FastMCP",
    b"McpServer",
    b"from mcp",
    b"import mcp",
    b"mcp.",
    b"ListToolsRequestSchema",
    b"CallToolRequestSchema",
    b".tool",
    b".registerTool",
    b".addTool",
    b".add_tool",
    b".setRequestHandler",
    b".list_tools",
    b".call_tool",
)

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".eggs", "vendor",
    "test", "tests", "__tests__", "fixtures",
    # Prompt/reference snapshots can contain copied SDK internals or examples
    # that look like tool registrations but are not this server's exposed tools.
    ".prompts", "ai_references",
}

JS_FUNCTION_TYPES = {
    "arrow_function",
    "function_expression",
    "function",
    "function_declaration",
    "method_definition",
}
PY_FUNCTION_TYPES = {"function_definition", "async_function_definition", "lambda"}
PY_PARAMETER_TYPES = {
    "identifier",
    "typed_parameter",
    "default_parameter",
    "typed_default_parameter",
}
DOCSTRING_ARG_SECTIONS = {"Args", "Arguments", "Parameters"}
DOCSTRING_END_SECTIONS = {"Returns", "Raises", "Yields", "Examples", "Example"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _child_of_type(node: Node, node_type: str) -> Optional[Node]:
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _child_of_types(node: Node, node_types: set[str] | tuple[str, ...]) -> Optional[Node]:
    for child in node.children:
        if child.type in node_types:
            return child
    return None


_SIMPLE_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0",
    "\\": "\\", "'": "'", '"': '"', "`": "`", "\n": "", "\r\n": "",
}


def _decode_escape(seq: str) -> str:
    """Decode one Python/JS escape sequence such as ``\\n`` or ``\\u0041``."""
    body = seq[1:]
    if body in _SIMPLE_ESCAPES:
        return _SIMPLE_ESCAPES[body]
    try:
        if body[:1] in ("x", "u", "U"):
            return chr(int(body[1:].strip("{}"), 16))
        if body.startswith("N{") and body.endswith("}"):
            return unicodedata.lookup(body[2:-1])
        if body[:1] in "01234567":
            return chr(int(body, 8))
    except (ValueError, OverflowError, KeyError):
        return seq
    # Unknown escapes: JS drops the backslash; keep that simpler behavior.
    return body


def _py_string_content(content: Node, raw: bool) -> str:
    """Rebuild a Python ``string_content`` node, decoding nested escapes."""
    data = content.text
    if raw or not content.children:
        return data.decode("utf-8", errors="replace")
    parts: list[str] = []
    pos = content.start_byte
    for child in content.children:
        parts.append(data[pos - content.start_byte:child.start_byte - content.start_byte]
                     .decode("utf-8", errors="replace"))
        text = _text(child)
        if child.type == "escape_sequence":
            parts.append(_decode_escape(text))
        elif child.type == "escape_interpolation":  # ``{{`` / ``}}`` in f-strings
            parts.append(text[0])
        else:
            parts.append(text)
        pos = child.end_byte
    parts.append(data[pos - content.start_byte:].decode("utf-8", errors="replace"))
    return "".join(parts)


def _string_value(node: Optional[Node]) -> Optional[str]:
    """Return the value of a string-valued expression, or None if it is not one.

    Handles escapes, Python f-strings / implicit concatenation, JS template
    strings, parenthesized strings, and JS ``+`` concatenation with at least one
    literal operand. Interpolated expressions are kept as their source text
    (``{expr}`` / ``${expr}``) since their runtime value is unknown statically.
    """
    if node is None:
        return None
    if node.type == "parenthesized_expression":
        named = _named_children(node)
        return _string_value(named[0]) if len(named) == 1 else None

    if node.type == "concatenated_string":
        parts = [_string_value(c) for c in node.children if c.is_named]
        return None if any(p is None for p in parts) else "".join(parts)

    if node.type == "binary_expression":
        op = node.child_by_field_name("operator")
        if not op or _text(op) != "+":
            return None
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if not left or not right:
            return None
        left_val = _string_value(left)
        right_val = _string_value(right)
        if left_val is None and right_val is None:
            return None
        return (
            (left_val if left_val is not None else f"${{{_text(left)}}}")
            + (right_val if right_val is not None else f"${{{_text(right)}}}")
        )

    if node.type not in ("string", "template_string"):
        return None

    parts: list[str] = []
    raw = False
    for c in node.children:
        if c.type == "string_start":
            raw = "r" in _text(c).lower()
        elif c.type == "string_content":
            parts.append(_py_string_content(c, raw))
        elif c.type == "string_fragment":
            parts.append(_text(c))
        elif c.type == "escape_sequence":
            parts.append(_decode_escape(_text(c)))
        elif c.type in ("interpolation", "template_substitution"):
            parts.append(_text(c))
    return "".join(parts)


def _property_key(node: Node) -> str:
    """Normalize an identifier or quoted object-property key."""
    return _string_value(node) or _text(node).strip("'\"")


def _find_kwarg_node(arg_list: Node, key: str) -> Optional[Node]:
    """Find a keyword_argument by name and return its value node (Python)."""
    for child in arg_list.children:
        if child.type == "keyword_argument":
            k = child.child_by_field_name("name")
            v = child.child_by_field_name("value")
            if k and _text(k) == key and v:
                return v
    return None


def _find_kwarg(arg_list: Node, key: str) -> Optional[str]:
    """Find a keyword_argument by name and return its string value (Python)."""
    return _string_value(_find_kwarg_node(arg_list, key))


def _find_obj_prop(obj_node: Node, key: str) -> Optional[str]:
    """Find a property in a JS/TS object literal and return its string value."""
    value = _find_object_pair_value(obj_node, key)
    return _string_value(value) if value else None


def _docstring(func_node: Node) -> Optional[str]:
    """Extract the docstring from a Python function_definition body."""
    body = func_node.child_by_field_name("body")
    if not body or body.child_count == 0:
        return None
    first = body.children[0]
    if first.type == "expression_statement" and first.child_count > 0:
        expr = first.children[0]
        if expr.type == "string":
            val = _string_value(expr)
            if val:
                return val.strip()
    return None


def _extract_field_descriptions(params_node: Node) -> list[dict]:
    """Extract Field(description=...) from Python function parameters."""
    args = []
    for p in params_node.children:
        if p.type in ("typed_default_parameter", "default_parameter"):
            param_name = _text(p.children[0])
            default = p.child_by_field_name("value") or (
                p.children[-1] if p.child_count > 2 else None
            )
            desc = None
            if default and default.type == "call":
                al = _py_argument_list(default)
                if al:
                    desc = _find_kwarg(al, "description")
            if desc:
                args.append({"name": param_name, "description": desc})
    return args


def _extract_docstring_arg_descriptions(docstring: Optional[str]) -> dict[str, str]:
    """Extract simple Google-style ``Args:`` descriptions from a docstring."""
    if not docstring:
        return {}

    descriptions: dict[str, str] = {}
    current_name: Optional[str] = None
    in_args = False

    for raw_line in docstring.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        section = stripped.rstrip(":")
        if section in DOCSTRING_ARG_SECTIONS:
            in_args = True
            current_name = None
            continue
        if in_args and section in DOCSTRING_END_SECTIONS:
            break
        if not in_args:
            continue

        match = re.match(r"^([A-Za-z_][\w]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if match:
            current_name = match.group(1)
            descriptions[current_name] = match.group(2).strip()
        elif current_name:
            descriptions[current_name] = f"{descriptions[current_name]} {stripped}".strip()

    return descriptions


def _python_parameter_name(param_node: Node) -> Optional[str]:
    for child in param_node.children:
        if child.type == "identifier":
            return _text(child)
    if param_node.type == "identifier":
        return _text(param_node)
    return None


def _extract_py_signature_arguments(func_node: Node) -> list[dict]:
    """Extract tool arguments from a Python function signature."""
    params_node = func_node.child_by_field_name("parameters")
    if not params_node:
        return []

    field_descriptions = {
        item["name"]: item.get("description", "")
        for item in _extract_field_descriptions(params_node)
        if item.get("name")
    }
    doc_descriptions = _extract_docstring_arg_descriptions(_docstring(func_node))

    args: list[dict] = []
    for param in params_node.children:
        if not param.is_named:
            continue
        if param.type not in PY_PARAMETER_TYPES:
            continue
        name = _python_parameter_name(param)
        if not name or name in {"self", "cls"}:
            continue
        args.append({
            "name": name,
            "description": field_descriptions.get(name) or doc_descriptions.get(name, ""),
        })
    return args


def _extract_zod_descriptions(obj_node: Node) -> list[dict]:
    """Extract z.xxx().describe("...") from a JS/TS object."""
    args = []
    for child in obj_node.children:
        if child.type != "pair":
            continue
        if child.child_count < 3:
            continue
        key_node = child.children[0]
        val_node = child.children[2]
        param_name = _property_key(key_node)
        desc = _find_describe_in_chain(val_node)
        if desc:
            args.append({"name": param_name, "description": desc})
    return args


def _find_describe_in_chain(node: Node) -> Optional[str]:
    """Walk a chained call expression looking for .describe("...")."""
    if node.type != "call_expression":
        return None
    callee = node.child_by_field_name("function")
    if callee is None or callee.type != "member_expression":
        return None
    if _js_member_prop_name(callee) == "describe":
        positional = _js_positional_args(_js_arguments(node))
        if positional:
            return _string_value(positional[0])
    obj = callee.child_by_field_name("object")
    return _find_describe_in_chain(obj) if obj is not None else None


def _loc(filepath: str, node: Node) -> str:
    """Format file:line location."""
    return f"{filepath}:{node.start_point[0] + 1}"


def _tool_metadata(
    name: Optional[str],
    description: Optional[str],
    arguments: list[dict],
    entry: str,
    entry_handler: Optional[list[str]] = None,
) -> dict:
    tool = {
        "tool_name": name or "",
        "tool_description": description or "",
        "arguments": arguments,
        "entry": entry,
    }
    if entry_handler:
        # Handler expression that could not be located in this file (imported
        # function, ``this.method``, ``obj.fn``); the slicer resolves it from the
        # registration site instead of slicing the enclosing function.
        tool["entry_handler"] = entry_handler
    return tool


def _named_children(node: Optional[Node]) -> list[Node]:
    if not node:
        return []
    return [c for c in node.children if c.is_named]


def _py_argument_list(call_node: Node) -> Optional[Node]:
    return _child_of_type(call_node, "argument_list")


def _js_arguments(call_node: Node) -> Optional[Node]:
    return call_node.child_by_field_name("arguments")


def _py_positional_args(arg_list: Optional[Node]) -> list[Node]:
    if not arg_list:
        return []
    return [c for c in arg_list.children if c.is_named and c.type != "keyword_argument"]


def _js_positional_args(args_node: Optional[Node]) -> list[Node]:
    if not args_node:
        return []
    return [c for c in args_node.children if c.is_named]


def _py_function_defs(root: Node) -> dict[str, Node]:
    defs: dict[str, Node] = {}
    for node in _walk(root):
        if node.type not in ("function_definition", "async_function_definition"):
            continue
        name_node = node.child_by_field_name("name")
        if name_node:
            defs[_text(name_node)] = node
    return defs


def _js_function_defs(root: Node) -> dict[str, Node]:
    defs: dict[str, Node] = {}
    for node in _walk(root):
        fname = None
        fnode = node
        if node.type == "function_declaration":
            nn = node.child_by_field_name("name")
            if nn:
                fname = _text(nn)
        elif node.type == "variable_declarator":
            name = node.child_by_field_name("name")
            value = node.child_by_field_name("value")
            if name and value and value.type in ("arrow_function", "function_expression", "function"):
                defs[_text(name)] = value
        elif node.type == "method_definition":
            nn = node.child_by_field_name("name")
            if nn:
                fname = _text(nn)
        if fname:
            defs[fname] = fnode
    return defs


def _resolve_python_entry(
    node: Optional[Node],
    filepath: str,
    function_defs: dict[str, Node],
    fallback: Node,
) -> tuple[str, Optional[str]]:
    """Return ``(entry, entry_handler)`` for a handler expression."""
    if node is None:
        return _loc(filepath, fallback), None
    if node.type in PY_FUNCTION_TYPES:
        return _loc(filepath, node), None
    if node.type == "identifier":
        target = function_defs.get(_text(node))
        if target is not None:
            return _loc(filepath, target), None
    if node.type in ("identifier", "attribute"):
        return _loc(filepath, fallback), [_text(node)]
    return _loc(filepath, fallback), None


def _resolve_js_entry(
    candidates: list[Node],
    filepath: str,
    function_defs: dict[str, Node],
    fallback: Node,
) -> tuple[str, Optional[list[str]]]:
    """Return ``(entry, entry_handler)`` for handler candidates (see ``_js_handler_candidates``)."""
    if not candidates:
        return _loc(filepath, fallback), None
    first = candidates[0]
    if first.type in JS_FUNCTION_TYPES:
        return _loc(filepath, first), None
    if first.type == "identifier" and len(candidates) == 1:
        target = function_defs.get(_text(first))
        if target is not None:
            return _loc(filepath, target), None
    return _loc(filepath, fallback), [_text(c) for c in candidates]


def _py_call_name(call_node: Node) -> str:
    func = call_node.child_by_field_name("function")
    if func is not None and func.type == "attribute":
        return _text(func.children[-1])
    return _text(func) if func is not None else ""


def _js_member_expression(node: Node) -> Optional[Node]:
    return node.child_by_field_name("function") or _child_of_type(node, "member_expression")


def _js_member_prop_name(node: Optional[Node]) -> str:
    if not node or node.type != "member_expression":
        return ""
    prop = node.child_by_field_name("property") or _child_of_type(node, "property_identifier")
    return _text(prop) if prop else ""


def _member_receiver_name(node: Optional[Node]) -> str:
    """Return the object on the left side of a Python/JS member access."""
    if not node:
        return ""
    obj = node.child_by_field_name("object")
    return _text(obj) if obj else ""


def _is_mcp_receiver(name: str, discovered: Optional[set[str]] = None) -> bool:
    """Recognize common names used for MCP server instances."""
    leaf = name.rsplit(".", 1)[-1].lower()
    return (
        name in (discovered or ())
        or leaf in {"app", "mcp", "registry", "self", "server", "srv", "this"}
        or "mcp" in leaf
        or leaf.endswith("server")
    )


def _discover_mcp_receivers(root: Node) -> set[str]:
    """Find variables initialized from a recognizable MCP server constructor."""
    receivers: set[str] = set()
    for node in _walk(root):
        if node.type in {"assignment", "variable_declarator"}:
            name = node.child_by_field_name("left") or node.child_by_field_name("name")
            value = node.child_by_field_name("right") or node.child_by_field_name("value")
        else:
            continue
        if not name or not value or value.type not in {"call", "call_expression", "new_expression"}:
            continue
        constructor = (
            value.child_by_field_name("function")
            or value.child_by_field_name("constructor")
        )
        constructor_name = _text(constructor).rsplit(".", 1)[-1].lower() if constructor else ""
        if "mcp" in constructor_name or constructor_name.endswith("server"):
            receivers.add(_text(name))
    return receivers


def _js_method_calls(root: Node, *method_names: str):
    """Yield JS/TS member calls and their positional arguments."""
    methods = set(method_names)
    for node in _walk(root):
        if node.type != "call_expression":
            continue
        if _js_member_prop_name(_js_member_expression(node)) not in methods:
            continue
        positional = _js_positional_args(_js_arguments(node))
        if positional:
            yield node, positional


_JS_TRANSPARENT_TYPES = {
    "as_expression", "satisfies_expression", "non_null_expression",
    "parenthesized_expression", "type_assertion",
}
_JS_NON_HANDLER_TYPES = {"string", "template_string", "object", "array", "number"}


def _js_handler_candidates(node: Optional[Node], depth: int = 0) -> list[Node]:
    """Nodes a handler expression may stand for, most likely first.

    An inline function is returned alone. Otherwise references are listed for
    the slicer to try in order. Unwraps casts (``handler as any``),
    ``fn.bind(this)``, and wrapper / factory calls: ``withErrorHandling(async
    (args) => ...)`` yields the inline function, ``wrap(handler)`` yields
    ``handler`` then ``wrap``, and ``createHandler(teamId)`` falls back to the
    factory ``createHandler`` whose returned closure is the handler.
    """
    if node is None or depth > 5:
        return []
    if node.type in JS_FUNCTION_TYPES or node.type in ("identifier", "member_expression"):
        return [node]
    if node.type in _JS_TRANSPARENT_TYPES:
        named = _named_children(node)
        return _js_handler_candidates(named[0], depth + 1) if named else []
    if node.type != "call_expression":
        return []
    callee = node.child_by_field_name("function")
    if callee is not None and _js_member_prop_name(callee) in ("bind", "call", "apply"):
        return _js_handler_candidates(callee.child_by_field_name("object"), depth + 1)
    args = _js_positional_args(_js_arguments(node))
    for arg in args:
        if arg.type in JS_FUNCTION_TYPES:
            return [arg]
    candidates: list[Node] = []
    for arg in reversed(args):
        candidates.extend(_js_handler_candidates(arg, depth + 1))
    for candidate in candidates:
        if candidate.type in JS_FUNCTION_TYPES:
            return [candidate]
    if callee is not None and callee.type in ("identifier", "member_expression"):
        candidates.append(callee)
    return candidates


def _last_js_handler_arg(args: list[Node]) -> list[Node]:
    """Handler candidates from registration arguments, last argument first.

    The handler is usually the last argument, but some wrappers take a trailing
    schema (``registerTool(name, opts, this.fn.bind(this), schemas.x)``), so
    references from every argument are kept for the slicer to try in order.
    """
    candidates: list[Node] = []
    for arg in reversed(args):
        if arg.type in _JS_NON_HANDLER_TYPES:
            continue
        candidates.extend(_js_handler_candidates(arg))
    for candidate in candidates:
        if candidate.type in JS_FUNCTION_TYPES:
            return [candidate]
    return candidates


def _find_object_pair_value(obj_node: Node, key: str) -> Optional[Node]:
    for child in obj_node.children:
        if child.type != "pair" or child.child_count < 3:
            continue
        k = child.child_by_field_name("key") or child.children[0]
        v = child.child_by_field_name("value") or child.children[2]
        if _property_key(k) == key:
            return v
    return None


def _py_decorated_functions(root: Node):
    for node in _walk(root):
        if node.type != "decorated_definition":
            continue
        decorators = [child for child in node.children if child.type == "decorator"]
        func_def = _child_of_types(node, ("function_definition", "async_function_definition"))
        if func_def:
            yield func_def, decorators


def _py_decorator_call(dec: Node) -> tuple[str, str, Optional[Node]]:
    call_node = _child_of_type(dec, "call")
    attr_node = _child_of_type(dec, "attribute")
    arg_list = None

    if call_node:
        attr_node = _child_of_type(call_node, "attribute") or attr_node
        arg_list = _py_argument_list(call_node)

    method_name = _text(attr_node.children[-1]) if attr_node and attr_node.children else ""
    return _member_receiver_name(attr_node), method_name, arg_list


# ---------------------------------------------------------------------------
# Pattern 1: @xxx.tool(...) in Python
# ---------------------------------------------------------------------------

def _extract_py_decorator_tool(root: Node, filepath: str) -> list[dict]:
    tools = []
    receivers = _discover_mcp_receivers(root)
    for func_def, decorators in _py_decorated_functions(root):
        for dec in decorators:
            receiver, method_name, arg_list = _py_decorator_call(dec)
            if method_name != "tool" or not _is_mcp_receiver(receiver, receivers):
                continue

            func_name_node = func_def.child_by_field_name("name")
            func_name = _text(func_name_node) if func_name_node else "unknown"

            tool_name = None
            tool_desc = None
            if arg_list:
                tool_name = _find_kwarg(arg_list, "name")
                tool_desc = _find_kwarg(arg_list, "description")
                positional = _py_positional_args(arg_list)
                if not tool_name and positional:
                    tool_name = _string_value(positional[0])

            if not tool_name:
                tool_name = func_name
            if not tool_desc:
                tool_desc = _docstring(func_def)

            arg_descs = _extract_py_signature_arguments(func_def)

            tools.append(
                _tool_metadata(tool_name, tool_desc, arg_descs, _loc(filepath, func_def))
            )
    return tools


# ---------------------------------------------------------------------------
# Pattern 2: xxx.add_tool() in Python
# ---------------------------------------------------------------------------

def _extract_py_add_tool(root: Node, filepath: str) -> list[dict]:
    tools = []
    function_defs = _py_function_defs(root)
    for node in _walk(root):
        if node.type != "call":
            continue
        if _py_call_name(node) != "add_tool":
            continue

        arg_list = _py_argument_list(node)
        if not arg_list:
            continue

        positional = _py_positional_args(arg_list)
        entry_node = positional[0] if positional else _find_kwarg_node(arg_list, "fn")
        func_ref = _text(entry_node) if entry_node is not None else "unknown"
        target_func = function_defs.get(func_ref)
        tool_name = _find_kwarg(arg_list, "name") or func_ref
        tool_desc = _find_kwarg(arg_list, "description")
        if not tool_desc and target_func is not None:
            tool_desc = _docstring(target_func)
        arguments = _extract_py_signature_arguments(target_func) if target_func is not None else []

        tools.append(
            _tool_metadata(
                tool_name,
                tool_desc,
                arguments,
                *_resolve_python_entry(entry_node, filepath, function_defs, node),
            )
        )
    return tools


def _extract_py_tool_metadata_from_expr(expr: Node, filepath: str) -> list[dict]:
    tools: list[dict] = []
    for node in _walk(expr):
        if node.type == "call":
            call_name = _py_call_name(node)
            if call_name != "Tool":
                continue
            arg_list = _py_argument_list(node)
            if not arg_list:
                continue
            tool_name = _find_kwarg(arg_list, "name") or ""
            if not tool_name:
                continue
            tool_desc = _find_kwarg(arg_list, "description") or ""
            input_schema = _find_kwarg(arg_list, "inputSchema")
            arguments = [{"name": "inputSchema", "description": input_schema}] if input_schema else []
            tools.append(
                _tool_metadata(tool_name, tool_desc, arguments, _loc(filepath, node))
            )
        elif node.type == "dictionary":
            name_val = None
            desc_val = ""
            for child in node.children:
                if child.type != "pair":
                    continue
                key = child.child_by_field_name("key") or (child.children[0] if child.children else None)
                value = child.child_by_field_name("value") or (child.children[-1] if child.children else None)
                if not key or not value:
                    continue
                key_text = _string_value(key) or _text(key).strip("'\"")
                if key_text == "name":
                    name_val = _string_value(value)
                elif key_text == "description":
                    desc_val = _string_value(value) or ""
            if name_val:
                tools.append(
                    _tool_metadata(name_val, desc_val, [], _loc(filepath, node))
                )
    return tools


def _extract_py_returned_tools(list_func: Node, filepath: str) -> list[dict]:
    tools: list[dict] = []
    body = list_func.child_by_field_name("body")
    if not body:
        return tools
    for node in _walk(body):
        if node.type == "return_statement":
            for value in _named_children(node)[:1]:
                tools.extend(_extract_py_tool_metadata_from_expr(value, filepath))
    return tools


def _py_dispatch_var_names(call_func: Node) -> set[str]:
    """Names that may hold the requested tool name inside a ``call_tool`` handler."""
    names = {"name", "tool_name"}
    params = call_func.child_by_field_name("parameters")
    for param in _named_children(params):
        if param.type not in PY_PARAMETER_TYPES:
            continue
        param_name = _python_parameter_name(param)
        if param_name in {"self", "cls"}:
            continue
        if param_name:
            names.add(param_name)
        break
    return names


def _py_string_literals(node: Optional[Node]) -> list[str]:
    """String values of a literal, or of each element of a tuple/list/set literal."""
    if node is None:
        return []
    value = _string_value(node)
    if value is not None:
        return [value]
    if node.type in ("tuple", "list", "set", "parenthesized_expression"):
        values = [_string_value(c) for c in _named_children(node)]
        if values and all(v is not None for v in values):
            return values
    return []


def _py_names_from_condition(node: Optional[Node], var_names: set[str]) -> list[str]:
    """Tool names tested by ``name == "x"``, ``"x" == name``, ``name in (...)``, or ``or`` of those."""
    if node is None:
        return []
    if node.type == "parenthesized_expression":
        named = _named_children(node)
        return _py_names_from_condition(named[0], var_names) if len(named) == 1 else []
    if node.type == "boolean_operator":
        op = node.child_by_field_name("operator")
        if op is None or _text(op) != "or":
            return []
        return (
            _py_names_from_condition(node.child_by_field_name("left"), var_names)
            + _py_names_from_condition(node.child_by_field_name("right"), var_names)
        )
    if node.type != "comparison_operator":
        return []
    named = _named_children(node)
    operators = [_text(c) for c in node.children if not c.is_named]
    if len(named) != 2 or len(operators) != 1:
        return []
    left, right = named
    if operators[0] == "==":
        if _text(left) in var_names:
            return _py_string_literals(right)[:1]
        if _text(right) in var_names:
            return _py_string_literals(left)[:1]
    elif operators[0] == "in" and _text(left) in var_names:
        return _py_string_literals(right)
    return []


def _py_dispatch_branches(body: Node, var_names: set[str]):
    """Yield ``(tool_name, block)`` for each if/elif/match-case branch keyed on the tool name."""
    for node in _walk(body):
        if node.type == "if_statement":
            clauses = [node] + [c for c in node.children if c.type == "elif_clause"]
            for clause in clauses:
                block = clause.child_by_field_name("consequence")
                if block is None:
                    continue
                for tool_name in _py_names_from_condition(
                    clause.child_by_field_name("condition"), var_names
                ):
                    yield tool_name, block
        elif node.type == "match_statement":
            subject = node.child_by_field_name("subject")
            if subject is None or _text(subject) not in var_names:
                continue
            for clause in _named_children(node.child_by_field_name("body")):
                if clause.type != "case_clause":
                    continue
                block = clause.child_by_field_name("consequence")
                pattern = _child_of_type(clause, "case_pattern")
                if block is None or pattern is None:
                    continue
                for alt in _named_children(pattern):
                    options = _named_children(alt) if alt.type == "union_pattern" else [alt]
                    for option in options:
                        tool_name = _string_value(option)
                        if tool_name:
                            yield tool_name, block


def _entry_block(nodes: list[Node]) -> dict:
    """Source range of a dispatch branch body, sliced instead of a whole function."""
    first, last = nodes[0], nodes[-1]
    return {
        "start_line": first.start_point[0] + 1,
        "end_line": last.end_point[0] + 1,
        "start_byte": first.start_byte,
        "end_byte": last.end_byte,
    }


def _apply_dispatch(tools: list[dict], dispatch: dict[str, tuple[str, dict]]) -> None:
    for tool in tools:
        hit = dispatch.get(tool["tool_name"])
        if hit:
            tool["entry"], tool["entry_block"] = hit


def _extract_py_dispatch_map(call_func: Node, filepath: str) -> dict[str, tuple[str, dict]]:
    """Map tool name -> (entry, entry_block) for each branch of a ``call_tool`` handler.

    The whole branch body becomes the slice root, so argument handling and
    validation in the branch are kept and other tools' branches are left out.
    """
    dispatch: dict[str, tuple[str, dict]] = {}
    body = call_func.child_by_field_name("body")
    if not body:
        return dispatch
    var_names = _py_dispatch_var_names(call_func)
    for tool_name, block in _py_dispatch_branches(body, var_names):
        if tool_name not in dispatch:  # keep the outermost / first branch
            dispatch[tool_name] = (_loc(filepath, block), _entry_block([block]))
    return dispatch


def _extract_py_list_call_dispatch(root: Node, filepath: str) -> list[dict]:
    list_func = None
    call_func = None
    for func_def, decorators in _py_decorated_functions(root):
        for dec in decorators:
            _, method_name, _ = _py_decorator_call(dec)
            if method_name == "list_tools":
                list_func = func_def
            elif method_name == "call_tool":
                call_func = func_def
    if not list_func:
        return []
    tools = _extract_py_returned_tools(list_func, filepath)
    dispatch_map = _extract_py_dispatch_map(call_func, filepath) if call_func else {}
    _apply_dispatch(tools, dispatch_map)
    # A single declared tool with no name-based branching: the whole call_tool
    # handler implements it.
    if call_func is not None and not dispatch_map and len(tools) == 1:
        tools[0]["entry"] = _loc(filepath, call_func)
    return tools


# ---------------------------------------------------------------------------
# Pattern 3: server.tool(...) in JS/TS
# ---------------------------------------------------------------------------

def _extract_js_server_tool(root: Node, filepath: str) -> list[dict]:
    tools = []
    function_defs = _js_function_defs(root)
    receivers = _discover_mcp_receivers(root)
    for node, positional in _js_method_calls(root, "tool"):
        if not _is_mcp_receiver(
            _member_receiver_name(_js_member_expression(node)), receivers
        ):
            continue

        tool_name = _string_value(positional[0]) or ""
        tool_desc = _string_value(positional[1]) if len(positional) > 1 else ""
        if not tool_desc:
            tool_desc = ""

        arg_descs = []
        schema_obj = None
        for p in positional:
            if p.type == "object":
                schema_obj = p
                break
        if schema_obj:
            arg_descs = _extract_zod_descriptions(schema_obj)

        # Handler is usually the last positional arg: inline function or a
        # reference like `translateText` (not covered by inline-only detection).
        handler_node = _last_js_handler_arg(positional)

        tools.append(
            _tool_metadata(
                tool_name,
                tool_desc,
                arg_descs,
                *_resolve_js_entry(handler_node, filepath, function_defs, node),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Pattern 4: server.registerTool(...) in JS/TS
# ---------------------------------------------------------------------------

def _extract_js_register_tool(root: Node, filepath: str) -> list[dict]:
    tools = []
    function_defs = _js_function_defs(root)
    for node, positional in _js_method_calls(root, "registerTool", "RegisterTool"):

        tool_name = _string_value(positional[0]) or ""
        tool_desc = ""
        arg_descs = []

        opts_obj = positional[1] if len(positional) > 1 and positional[1].type == "object" else None
        if opts_obj:
            tool_desc = _find_obj_prop(opts_obj, "description") or ""
            input_schema = _find_object_pair_value(opts_obj, "inputSchema")
            if input_schema and input_schema.type == "object":
                arg_descs = _extract_zod_descriptions(input_schema)

        handler_node = _last_js_handler_arg(positional)

        tools.append(
            _tool_metadata(
                tool_name,
                tool_desc,
                arg_descs,
                *_resolve_js_entry(handler_node, filepath, function_defs, node),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Pattern 5: xxx.addTool({name, description, ...}, handler) in JS/TS
# ---------------------------------------------------------------------------

def _extract_js_add_tool(root: Node, filepath: str) -> list[dict]:
    tools = []
    function_defs = _js_function_defs(root)
    for node, positional in _js_method_calls(root, "addTool"):

        opts_obj = positional[0] if positional[0].type == "object" else None
        if not opts_obj:
            continue

        tool_name = _find_obj_prop(opts_obj, "name") or ""
        tool_desc = _find_obj_prop(opts_obj, "description") or ""

        arg_descs = []
        for schema_key in ("inputSchema", "parameters", "schema"):
            schema = _find_object_pair_value(opts_obj, schema_key)
            if schema and schema.type == "object":
                arg_descs = _extract_json_schema_args(schema)
                break

        handler_node = _last_js_handler_arg(positional[1:])
        if not handler_node:
            # FastMCP (TS) style: addTool({ name, description, parameters, execute })
            for key in ("execute", "handler", "callback", "run"):
                handler_node = _js_handler_candidates(_find_object_pair_value(opts_obj, key))
                if handler_node:
                    break

        tools.append(
            _tool_metadata(
                tool_name,
                tool_desc,
                arg_descs,
                *_resolve_js_entry(handler_node, filepath, function_defs, node),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Pattern 6: setRequestHandler(ListToolsRequestSchema/CallToolRequestSchema)
# ---------------------------------------------------------------------------

def _extract_js_tool_metadata_from_node(node: Node, filepath: str) -> list[dict]:
    tools = []
    stack = [node]
    while stack:
        obj = stack.pop()
        name_val = _find_obj_prop(obj, "name") if obj.type == "object" else None
        if not name_val:
            stack.extend(reversed(obj.children))
            continue
        # Objects nested in a tool definition (schema properties, parameter lists,
        # annotations) may also carry name/description; they are not tools.
        desc_val = _find_obj_prop(obj, "description")
        arg_descs = []
        input_schema = _find_object_pair_value(obj, "inputSchema")
        if input_schema and input_schema.type == "object":
            arg_descs = _extract_json_schema_args(input_schema)

        tools.append(
            _tool_metadata(name_val, desc_val, arg_descs, _loc(filepath, obj))
        )
    return tools


def _extract_js_request_handler_calls(root: Node) -> tuple[list[Node], list[Node]]:
    list_calls: list[Node] = []
    call_calls: list[Node] = []
    for node, positional in _js_method_calls(root, "setRequestHandler"):
        first_text = _text(positional[0])
        if "ListToolsRequestSchema" in first_text or "ListTool" in first_text:
            list_calls.append(node)
        elif "CallToolRequestSchema" in first_text or "CallTool" in first_text:
            call_calls.append(node)
    return list_calls, call_calls


def _js_extract_handler_function(
    call_node: Node,
    function_defs: dict[str, Node],
) -> Optional[Node]:
    positional = _js_positional_args(_js_arguments(call_node))
    if len(positional) < 2:
        return None
    handler = positional[1]
    if handler.type in ("arrow_function", "function_expression", "function"):
        return handler
    if handler.type == "identifier":
        return function_defs.get(_text(handler))
    return None


def _js_names_from_condition(condition: Node) -> list[str]:
    """Tool names compared with ``==`` / ``===`` in an if condition (incl. ``||`` chains)."""
    names: list[str] = []
    for node in _walk(condition):
        if node.type != "binary_expression":
            continue
        op = node.child_by_field_name("operator")
        if op is None or _text(op) not in ("===", "=="):
            continue
        for side in ("left", "right"):
            value = _string_value(node.child_by_field_name(side))
            if value is not None:
                names.append(value)
                break
    if names:
        return names
    # Unusual conditions: fall back to the first string literal.
    for node in _walk(condition):
        value = _string_value(node)
        if value is not None:
            return [value]
    return []


def _js_dispatch_branches(handler: Node):
    """Yield ``(tool_name, body_nodes)`` for if / switch-case branches in a CallTool handler."""
    for node in _walk(handler):
        if node.type == "if_statement":
            condition = node.child_by_field_name("condition")
            consequence = node.child_by_field_name("consequence")
            if condition is None or consequence is None:
                continue
            for tool_name in _js_names_from_condition(condition):
                yield tool_name, [consequence]
        elif node.type == "switch_statement":
            pending: list[str] = []
            for case in _named_children(node.child_by_field_name("body")):
                if case.type == "switch_case":
                    value = _string_value(case.child_by_field_name("value"))
                    if value is not None:
                        pending.append(value)
                elif case.type != "switch_default":
                    continue
                statements = case.children_by_field_name("body")
                if statements:  # empty cases fall through to the next body
                    for tool_name in pending:
                        yield tool_name, statements
                    pending = []


def _extract_js_call_dispatch_map(
    call_node: Node,
    filepath: str,
    function_defs: dict[str, Node],
) -> dict[str, tuple[str, dict]]:
    dispatch: dict[str, tuple[str, dict]] = {}
    handler = _js_extract_handler_function(call_node, function_defs)
    if not handler:
        return dispatch
    for tool_name, body in _js_dispatch_branches(handler):
        if tool_name not in dispatch:  # keep the outermost / first branch
            dispatch[tool_name] = (_loc(filepath, body[0]), _entry_block(body))
    return dispatch


def _extract_js_list_call_dispatch(root: Node, filepath: str) -> list[dict]:
    function_defs = _js_function_defs(root)
    list_calls, call_calls = _extract_js_request_handler_calls(root)
    tools: list[dict] = []
    for node in list_calls:
        handler = _js_extract_handler_function(node, function_defs)
        tools.extend(_extract_js_tool_metadata_from_node(handler or node, filepath))
    dispatch: dict[str, tuple[str, dict]] = {}
    for node in call_calls:
        for tool_name, hit in _extract_js_call_dispatch_map(node, filepath, function_defs).items():
            dispatch.setdefault(tool_name, hit)
    _apply_dispatch(tools, dispatch)
    # A single declared tool with no name-based branching: the whole CallTool
    # handler implements it.
    if not dispatch and len(tools) == 1 and len(call_calls) == 1:
        handler = _js_extract_handler_function(call_calls[0], function_defs)
        if handler is not None:
            tools[0]["entry"] = _loc(filepath, handler)
    return tools


def _extract_json_schema_args(schema_obj: Node) -> list[dict]:
    """Extract argument descriptions from a JSON Schema 'properties' object."""
    args = []
    props_obj = None
    for child in schema_obj.children:
        if child.type == "pair" and child.child_count >= 3:
            k = child.children[0]
            v = child.children[2]
            if _property_key(k) == "properties" and v.type == "object":
                props_obj = v
                break
    if not props_obj:
        return args

    for child in props_obj.children:
        if child.type != "pair" or child.child_count < 3:
            continue
        param_name = _property_key(child.children[0])
        val_obj = child.children[2]
        if val_obj.type == "object":
            desc = _find_obj_prop(val_obj, "description")
            if desc:
                args.append({"name": param_name, "description": desc})
    return args


# ---------------------------------------------------------------------------
# Tree walk
# ---------------------------------------------------------------------------

def _walk(node: Node):
    """Depth-first walk yielding all nodes (iterative to avoid RecursionError)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def _deduplicate_tools_by_name(tools: list[dict]) -> list[dict]:
    """Keep the first concrete tool for each exposed name within one server."""
    seen: set[str] = set()
    result: list[dict] = []
    for tool in tools:
        name = (tool.get("tool_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(tool)
    return result


def _has_required_tool_metadata(tool: dict) -> bool:
    """Keep only concrete tools with both registry name and description.

    Empty fields usually mean one of two things: the server truly omitted the
    tool description, or the registration is dynamic/wrapper-based and cannot be
    resolved statically from the registry call. In both cases the tool is not a
    valid description-vs-code auditing target for this stage.
    """
    return bool((tool.get("tool_name") or "").strip()) and bool(
        (tool.get("tool_description") or "").strip()
    )


def _filter_tools_with_metadata(tools: list[dict]) -> list[dict]:
    return [tool for tool in tools if _has_required_tool_metadata(tool)]


def _scan_file(filepath: str) -> list[dict]:
    ext = Path(filepath).suffix.lower()
    entry = LANG_MAP.get(ext)
    if not entry:
        return []

    lang_name, parser = entry

    try:
        source = Path(filepath).read_bytes()
    except OSError:
        return []

    if len(source) > MAX_SOURCE_BYTES:
        return []

    if not any(marker in source for marker in MCP_SOURCE_MARKERS):
        return []

    try:
        tree = parser.parse(source)
    except Exception:
        return []

    root = tree.root_node
    abs_path = str(Path(filepath).resolve())
    tools: list[dict] = []

    if lang_name == "python":
        tools.extend(_extract_py_decorator_tool(root, abs_path))
        tools.extend(_extract_py_add_tool(root, abs_path))
        tools.extend(_extract_py_list_call_dispatch(root, abs_path))
    else:
        tools.extend(_extract_js_server_tool(root, abs_path))
        tools.extend(_extract_js_register_tool(root, abs_path))
        tools.extend(_extract_js_add_tool(root, abs_path))
        tools.extend(_extract_js_list_call_dispatch(root, abs_path))

    return _filter_tools_with_metadata(tools)


def scan_repo(repo_dir: str) -> list[dict]:
    """Scan all source files in a repo directory for MCP tool definitions."""
    all_tools = []
    repo_path = Path(repo_dir)

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext not in LANG_MAP:
                continue
            full = os.path.join(dirpath, fname)
            tools = _scan_file(full)
            all_tools.extend(tools)

    return _deduplicate_tools_by_name(all_tools)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _discover_server_dirs(root: Path) -> list[Path]:
    """Treat every immediate subdirectory of the source root as one MCP server."""
    root = root.resolve()
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name not in SKIP_DIRS
    )


def _failure_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}")


def _write_extract_payload(output_path: Path, results: list[dict], failures: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server_count": len(results),
        "tool_count": sum(len(r["tools"]) for r in results),
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        failure_payload = {
            "extracted_at": payload["extracted_at"],
            "server_count": len(failures),
            "results": failures,
        }
        _failure_output_path(output_path).write_text(
            json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        _failure_output_path(output_path).unlink(missing_ok=True)


def extract_tools(source_root: Path, output_path: Path) -> list[dict]:
    """Extract tools from MCP server directories directly under source_root."""
    source_root = source_root.resolve()

    server_dirs = _discover_server_dirs(source_root)
    logger.info("Discovered %d MCP server directories under %s", len(server_dirs), source_root)

    results = []
    failures = []
    total = len(server_dirs)

    for i, server_dir in enumerate(server_dirs, 1):
        name = server_dir.name

        logger.info("[%d/%d] Processing MCP server: %s", i, total, server_dir)
        tools = scan_repo(str(server_dir))
        if not tools:
            failures.append({
                "name": name,
                "local_repo_dir": str(server_dir),
                "reason": "no tool definitions found",
            })
            logger.info("  Found 0 tools")
            continue

        results.append({
            "name": name,
            "local_repo_dir": str(server_dir),
            "tools": tools,
            "total_tools_found": len(tools),
        })
        logger.info("  Found %d tools", len(tools))

    _write_extract_payload(output_path, results, failures)
    logger.info(
        "Extraction done: %d matched servers, %d failures → %s",
        len(results),
        len(failures),
        output_path,
    )
    return results
