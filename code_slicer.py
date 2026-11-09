"""
Unified Code Slicer: extract code slices from Python / TS / JS projects
starting from an entry location (file:line).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import textwrap
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import tree_sitter_javascript as tsjs
import tree_sitter_python as tspy
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser, Tree

logger = logging.getLogger(__name__)

PY_LANG = Language(tspy.language())
JS_LANG = Language(tsjs.language())
TS_LANG = Language(tsts.language_typescript())
TSX_LANG = Language(tsts.language_tsx())

PY_FUNCTION_TYPES = {"function_definition", "async_function_definition"}
JS_FUNCTION_TYPES = {
    "function_declaration",
    "method_definition",
    "arrow_function",
    "function_expression",
    "function",
}
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".cjs", ".jsx", ".mjs", ".ts", ".cts", ".mts", ".tsx",
}

# Line budget per slice (~80k chars, ~26k tokens); deeper callees beyond it are
# omitted and listed in the slice.
MAX_SLICE_LINES = 2000
# Per-tool slicing time limit (seconds).
SLICE_TIMEOUT_SEC = 120


class SliceTimeout(BaseException):
    """Raised when slicing one tool exceeds its time limit.

    Derives from BaseException so the ``except Exception`` in
    ``generate_slice_for_tool`` cannot swallow it.
    """


@contextmanager
def _time_limit(seconds: Optional[float]):
    """SIGALRM-based limit; a no-op when disabled or off the main thread."""
    if not seconds or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, frame):
        raise SliceTimeout(f"slicing timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

# ─────────────────────────────────────────────────────────
# Base slicer with shared logic
# ─────────────────────────────────────────────────────────

class _BaseSlicer:
    def __init__(self, project_root: str, parser: Parser):
        self.project_root = os.path.abspath(project_root)
        self.parser = parser
        self.file_cache: dict[str, tuple[Tree, bytes]] = {}
        self.functions: dict[str, dict[str, str]] = {}
        self.function_defs: dict[str, dict[str, tuple[int, int, Node]]] = {}
        self.function_locations: dict[str, list[tuple[str, int, int, Node]]] = {}
        self.visited: set[str] = set()
        self._names_by_range: dict[str, dict[tuple[int, int], str]] = {}
        self.func_id_counter = 0
        self.func_id_map: dict[tuple[str, str], str] = {}
        self.call_edges: dict[str, set[str]] = {}
        # qualified class name -> class node
        self.class_nodes: dict[str, dict[str, Node]] = {}

    # ── helpers ──

    def _get_func_id(self, filepath: str, func_name: str) -> str:
        key = (filepath, func_name)
        if key not in self.func_id_map:
            self.func_id_map[key] = f"func_{self.func_id_counter}"
            self.func_id_counter += 1
        return self.func_id_map[key]

    def _parse_file(self, filepath: str) -> tuple[Tree, bytes]:
        if filepath in self.file_cache:
            return self.file_cache[filepath]
        try:
            source = Path(filepath).read_bytes()
        except OSError:
            raise FileNotFoundError(filepath)
        tree = self._parser_for_file(filepath).parse(source)
        self.file_cache[filepath] = (tree, source)
        self._index_file(filepath, tree)
        return tree, source

    def _parser_for_file(self, filepath: str) -> Parser:
        return self.parser

    def _index_file(self, filepath: str, tree: Tree):
        raise NotImplementedError

    @staticmethod
    def _traverse(node: Node):
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    @staticmethod
    def _traverse_function_body(node: Node, function_types: set[str]):
        """Walk one function without attributing nested-function calls to it."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current is not node and current.type in function_types:
                continue
            yield current
            stack.extend(reversed(current.children))

    def _txt(self, node: Node) -> str:
        return node.text.decode("utf-8", errors="replace")

    def _content(self, filepath: str, node: Node) -> str:
        return self._source_text(filepath, node.start_byte, node.end_byte)

    # ── definition lookup (desc-gen ``find_definition`` tool) ──

    def find_definitions(self, filepath: str, symbol: str) -> list[tuple[str, str, Node]]:
        """Definitions in *filepath* named *symbol*, as ``(kind, name, node)``.

        A qualified *symbol* (``Client.send``) matches exactly; any symbol also
        matches as the last segments of a qualified name, so ``send`` finds the
        method ``Client.send``. Kinds are ``function``, ``class`` and ``variable``
        (module-level assignments).
        """
        self._parse_file(filepath)

        def matches(name: str) -> bool:
            return name == symbol or name.endswith("." + symbol)

        found: list[tuple[str, str, Node]] = []
        seen: set[tuple[int, int]] = set()

        def add(kind: str, name: str, node: Node) -> None:
            key = (node.start_byte, node.end_byte)
            if key not in seen:
                seen.add(key)
                found.append((kind, name, node))

        for name, _, _, node in self.function_locations.get(filepath, []):
            if matches(name):
                add("function", name, node)
        for name, node in self.class_nodes.get(filepath, {}).items():
            if matches(name):
                add("class", name, node)
        found_names = {name for _, name, _ in found}
        for name, node in self._top_level_assignments(filepath):
            # ``const f = () => ...`` is already listed as a function.
            if matches(name) and name not in found_names:
                add("variable", name, node)
        return sorted(found, key=lambda item: item[2].start_byte)

    def _top_level_assignments(self, filepath: str):
        """Yield ``(name, statement node)`` for module-level variable assignments."""
        return iter(())

    # ── comment / docstring removal ──

    COMMENT_TYPES = {"comment", "html_comment"}

    def _annotation_replacement(self, node: Node) -> Optional[bytes]:
        """Bytes to put in place of *node* if it is a comment or docstring, else None.

        Comments and docstrings are the author's claims about the code, not its
        behavior, so they are kept out of slices.
        """
        return b"" if node.type in self.COMMENT_TYPES else None

    def _source_text(self, filepath: str, start: int, end: int) -> str:
        """Source in ``[start, end)`` with comments and docstrings removed.

        Lines left blank by a removal are dropped and trailing whitespace before a
        removed end-of-line comment is trimmed; other lines are kept verbatim.
        """
        tree, src = self.file_cache[filepath]
        removals: list[tuple[int, int, bytes]] = []
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.end_byte <= start or node.start_byte >= end:
                continue
            replacement = self._annotation_replacement(node)
            if replacement is not None:
                removals.append((max(node.start_byte, start), min(node.end_byte, end), replacement))
                continue
            stack.extend(node.children)
        if not removals:
            return src[start:end].decode("utf-8", errors="replace")

        # A NUL marks each line a removal touched, so it can be trimmed or dropped.
        pieces: list[bytes] = []
        pos = start
        for s, e, replacement in sorted(removals):
            pieces.append(src[pos:s])
            pieces.append(replacement + b"\0")
            pos = e
        pieces.append(src[pos:end])
        lines: list[str] = []
        for line in b"".join(pieces).decode("utf-8", errors="replace").split("\n"):
            if "\0" in line:
                line = line.replace("\0", "").rstrip()
                if not line.strip():
                    continue
            lines.append(line)
        return "\n".join(lines)

    # ── entry ──

    def _find_containing_function(self, filepath: str, line: int):
        self._parse_file(filepath)
        best = None
        for name, sl, el, nd in self.function_locations.get(filepath, []):
            if sl <= line <= el:
                if best is None or (el - sl) < (best[2] - best[1]):
                    best = (name, sl, el, nd)
        return best

    def _find_nearest_function(self, filepath: str, line: int):
        self._parse_file(filepath)
        nearest = None
        for name, sl, el, nd in self.function_locations.get(filepath, []):
            if sl >= line and (nearest is None or sl < nearest[1]):
                nearest = (name, sl, el, nd)
        if nearest and nearest[1] - line < 150:
            return nearest
        return None

    # ── callee extraction (language-specific) ──

    def _extract_callees(self, func_node: Node) -> list[str]:
        raise NotImplementedError

    def _resolve_callee(self, filepath: str, callee_name: str, func_name: str = None):
        raise NotImplementedError

    # ── classes (language hooks: _resolve_class, _class_bases) ──

    CONSTRUCTOR = ""
    EXCEPTION_RE = re.compile(r"(Error|Exception)$")

    def _resolve_class(self, filepath: str, name: str) -> Optional[tuple[str, str]]:
        """Locate class *name* as seen from *filepath*: ``(file, qualified class name)``."""
        raise NotImplementedError

    def _class_bases(self, filepath: str, qualified: str) -> list[str]:
        raise NotImplementedError

    def _is_exception_class(self, filepath: str, qualified: str, depth: int = 0) -> bool:
        if self.EXCEPTION_RE.search(qualified):
            return True
        if depth > 3:
            return False
        for base in self._class_bases(filepath, qualified):
            if self.EXCEPTION_RE.search(base):
                return True
            loc = self._resolve_class(filepath, base)
            if loc and self._is_exception_class(*loc, depth + 1):
                return True
        return False

    def _class_member(
        self, filepath: str, class_name: str, member: str, depth: int = 0
    ) -> Optional[tuple[str, str]]:
        """Resolve ``class_name.member`` (including project base classes)."""
        loc = self._resolve_class(filepath, class_name)
        if not loc or depth > 3:
            return None
        cf, qualified = loc
        key = f"{qualified}.{member}"
        if key in self.function_defs.get(cf, {}):
            return (cf, key)
        for base in self._class_bases(cf, qualified):
            found = self._class_member(cf, base, member, depth + 1)
            if found:
                return found
        return None

    def _constructor(self, filepath: str, class_name: str) -> Optional[tuple[str, str]]:
        """``Foo(...)`` / ``new Foo(...)`` on a project class -> its constructor (exceptions skipped)."""
        loc = self._resolve_class(filepath, class_name)
        if not loc or self._is_exception_class(*loc):
            return None
        return self._class_member(filepath, class_name, self.CONSTRUCTOR)

    def _owner_class(self, filepath: str, func_name: Optional[str]) -> Optional[str]:
        """Innermost class that *func_name* (``Cls.method`` / ``Cls.[inline:N]``) belongs to."""
        parts = (func_name or "").split(".")
        classes = self.class_nodes.get(filepath, {})
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in classes:
                return candidate
        return None

    @staticmethod
    def _unique_instance_class(instances: dict[tuple[str, str], str], var: str) -> Optional[str]:
        """The class *var* holds when every assignment in the file agrees."""
        classes = {cls for (_, name), cls in instances.items() if name == var}
        return classes.pop() if len(classes) == 1 else None

    # ── recursive analysis ──

    def _reset_analysis(self) -> None:
        self.functions.clear()
        self.visited.clear()
        self.func_id_counter = 0
        self.func_id_map.clear()
        self.call_edges: dict[str, set[str]] = {}

    def _analyze_function(self, filepath: str, func_name: str, func_node: Node):
        func_id = self._get_func_id(filepath, func_name)
        if func_id in self.visited:
            return
        self.visited.add(func_id)

        content = self._content(filepath, func_node)
        rel = os.path.relpath(filepath, self.project_root)
        full_name = f"{rel}::{func_name}"

        self._follow_callees(filepath, func_name, [func_node], caller_id=func_id)
        self.functions[func_id] = {"func_name": full_name, "content": content}

    FUNCTION_TYPES: set[str] = set()

    def _follow_callees(
        self,
        filepath: str,
        context_name: str,
        nodes: list[Node],
        caller_id: Optional[str] = None,
    ) -> None:
        """Resolve calls made inside *nodes* (scoped as *context_name*) and analyze them.

        Nested functions inside *nodes* (callbacks such as ``items.map(x => f(x))``,
        ``new Promise(...)`` executors, closures returned by factories, inner
        ``def``s) are part of the sliced source text, so their calls are followed
        too, each resolved in the nested function's own scope.
        """
        for node in nodes:
            for callee_name in self._extract_callees(node):
                resolved = self._resolve_callee(filepath, callee_name, context_name)
                if not resolved:
                    continue
                cf, cn = resolved
                info = self.function_defs.get(cf, {}).get(cn)
                if info:
                    if caller_id is not None:
                        self.call_edges.setdefault(caller_id, set()).add(self._get_func_id(cf, cn))
                    self._analyze_function(cf, cn, info[2])
            for nested in self._nested_functions(node):
                self._follow_callees(
                    filepath,
                    self._function_name_at(filepath, nested) or context_name,
                    [nested],
                    caller_id=caller_id,
                )

    def _nested_functions(self, node: Node):
        """Function nodes directly nested in *node* (not inside a deeper nested function)."""
        stack = list(reversed(node.children))
        while stack:
            current = stack.pop()
            if current.type in self.FUNCTION_TYPES:
                yield current
                continue
            stack.extend(reversed(current.children))

    def _function_name_at(self, filepath: str, node: Node) -> Optional[str]:
        """Indexed name of a function node (``outer.inner``, ``Cls.[inline:12]``), if any."""
        names = self._names_by_range.get(filepath)
        if names is None:
            names = self._names_by_range[filepath] = {
                (n.start_byte, n.end_byte): name
                for name, _, _, n in self.function_locations.get(filepath, [])
            }
        return names.get((node.start_byte, node.end_byte))

    def _resolve_entry(self, filepath: str, line: int):
        """Function containing *line*, else the nearest one below it."""
        hit = self._find_containing_function(filepath, line) or self._find_nearest_function(filepath, line)
        return (filepath, *hit) if hit else None

    def _analysis_result(self) -> dict[str, dict[str, str]]:
        """Analyzed functions in discovery order, each with its call depth from the entry."""
        order = sorted(self.functions, key=lambda value: int(value.split("_")[1]))
        depth: dict[str, int] = {}
        if order:
            frontier = [order[0]]
            depth[order[0]] = 0
            while frontier:  # BFS: shortest call distance from the entry
                nxt = []
                for fid in frontier:
                    for callee in sorted(self.call_edges.get(fid, ())):
                        if callee not in depth:
                            depth[callee] = depth[fid] + 1
                            nxt.append(callee)
                frontier = nxt
        return {
            fid: {**self.functions[fid], "depth": depth.get(fid, len(order))}
            for fid in order
        }

    def analyze(
        self,
        entry_file: str,
        entry_line: int,
        handler: Optional[str | list[str]] = None,
    ) -> dict[str, dict[str, str]]:
        self._reset_analysis()
        entry_file = os.path.abspath(entry_file)
        self._parse_file(entry_file)

        candidates = [handler] if isinstance(handler, str) else list(handler or [])
        if candidates:
            # The entry line is the registration call; resolve the handler
            # expression (imported name, ``this.method``, ``obj.fn``, wrapper or
            # factory) from there rather than slicing the function that performs
            # the registration. Candidates are tried in order.
            context = self._find_containing_function(entry_file, entry_line)
            context_name = context[0] if context else ""
            for candidate in candidates:
                resolved = self._resolve_callee(entry_file, candidate, context_name)
                info = self.function_defs.get(resolved[0], {}).get(resolved[1]) if resolved else None
                if info:
                    self._analyze_function(resolved[0], resolved[1], info[2])
                    return self._analysis_result()
            logger.debug("  Handler %s not resolved; falling back to line %d", candidates, entry_line)

        hit = self._resolve_entry(entry_file, entry_line)
        if not hit:
            raise ValueError(f"No function at line {entry_line} in {entry_file}")
        func_file, func_name, _, _, func_node = hit

        self._analyze_function(func_file, func_name, func_node)
        return self._analysis_result()

    def analyze_block(
        self,
        entry_file: str,
        start_byte: int,
        end_byte: int,
        label: str,
        expected_start_line: Optional[int] = None,
    ) -> dict[str, dict[str, str]]:
        """Slice from one dispatch branch body instead of a whole function.

        Used for low-level ``call_tool`` / ``CallToolRequestSchema`` handlers where
        every tool shares one dispatcher: only the branch for this tool and the
        functions it reaches are kept.
        """
        self._reset_analysis()
        entry_file = os.path.abspath(entry_file)
        tree, src = self._parse_file(entry_file)
        if not (0 <= start_byte < end_byte <= len(src)):
            raise ValueError(f"Branch range {start_byte}-{end_byte} is outside {entry_file}")
        start_line = src.count(b"\n", 0, start_byte) + 1
        if expected_start_line is not None and start_line != expected_start_line:
            raise ValueError(
                f"Branch range no longer matches {entry_file} "
                f"(line {start_line} != {expected_start_line}); re-run --extract"
            )

        container = tree.root_node.descendant_for_byte_range(start_byte, end_byte)
        if container.start_byte == start_byte and container.end_byte == end_byte:
            roots = [container]
        else:  # several statements, e.g. the body of a JS switch case
            roots = [
                c for c in container.children
                if c.start_byte >= start_byte and c.end_byte <= end_byte
            ]

        hit = self._find_containing_function(entry_file, start_line)
        context_name = hit[0] if hit else ""
        root_id = self._get_func_id(entry_file, f"{context_name}[branch:{label}]")
        self.visited.add(root_id)
        self._follow_callees(entry_file, context_name, roots, caller_id=root_id)

        rel = os.path.relpath(entry_file, self.project_root)
        self.functions[root_id] = {
            "func_name": f"{rel}::{context_name or '<module>'} [branch: {label}]",
            "content": self._block_text(entry_file, start_byte, end_byte),
        }
        return self._analysis_result()

    def _block_text(self, filepath: str, start_byte: int, end_byte: int) -> str:
        """Branch source (comments removed) with its common indentation removed."""
        _, src = self.file_cache[filepath]
        line_start = src.rfind(b"\n", 0, start_byte) + 1
        if not src[line_start:start_byte].strip():
            return textwrap.dedent(self._source_text(filepath, line_start, end_byte))
        # The branch starts mid-line (e.g. ``case "x": {``): keep the first line as is.
        first, _, rest = self._source_text(filepath, start_byte, end_byte).partition("\n")
        return f"{first}\n{textwrap.dedent(rest)}" if rest else first


# ─────────────────────────────────────────────────────────
# Python slicer
# ─────────────────────────────────────────────────────────

class PythonSlicer(_BaseSlicer):
    FUNCTION_TYPES = PY_FUNCTION_TYPES
    CONSTRUCTOR = "__init__"
    EXCEPTION_RE = re.compile(r"(Error|Exception|Warning)$")

    def __init__(self, project_root: str):
        super().__init__(project_root, Parser(PY_LANG))
        self.import_mappings: dict[str, dict[str, tuple[str, str]]] = {}
        # (scope function, variable) -> class expression, e.g. ("main", "client") -> "Client"
        self.instance_map: dict[str, dict[tuple[str, str], str]] = {}
        # qualified class name -> field -> class expression (``self.x = Foo()``)
        self.field_types: dict[str, dict[str, dict[str, str]]] = {}

    def _index_file(self, filepath, tree):
        self._extract_imports(filepath, tree)
        self._extract_function_defs(filepath, tree)
        self._extract_class_nodes(filepath, tree)
        self._extract_instance_map(filepath, tree)

    def _top_level_assignments(self, filepath: str):
        tree, _ = self.file_cache[filepath]
        for stmt in tree.root_node.children:
            if stmt.type != "expression_statement":
                continue
            for child in stmt.named_children:
                left = child.child_by_field_name("left") if child.type == "assignment" else None
                if left is not None and left.type == "identifier":
                    yield self._txt(left), stmt

    def _annotation_replacement(self, node: Node) -> Optional[bytes]:
        """Comments, plus docstrings and other bare string statements.

        A bare string statement has no effect, so it is dropped; f-strings with
        interpolations are kept because evaluating them can run code. A block
        whose only statement is removed gets ``pass`` so it stays valid Python.
        """
        if node.type in self.COMMENT_TYPES:
            return b""
        if node.type != "expression_statement" or node.named_child_count != 1:
            return None
        value = node.named_children[0]
        if value.type not in {"string", "concatenated_string"}:
            return None
        if any(n.type == "interpolation" for n in self._traverse(value)):
            return None
        parent = node.parent
        if parent is not None and parent.type == "block" and all(
            sibling == node or sibling.type in self.COMMENT_TYPES
            for sibling in parent.named_children
        ):
            return b"pass"
        return b""

    def _function_name(self, node: Node) -> str:
        parts = [self._txt(node.child_by_field_name("name"))]
        parent = node.parent
        while parent:
            if parent.type in PY_FUNCTION_TYPES or parent.type == "class_definition":
                owner = parent.child_by_field_name("name")
                if owner:
                    parts.append(self._txt(owner))
            parent = parent.parent
        return ".".join(reversed(parts))

    def _extract_imports(self, filepath, tree):
        self.import_mappings[filepath] = {}
        for node in self._traverse(tree.root_node):
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        mod = self._txt(child)
                        self.import_mappings[filepath][mod] = (mod, mod)
                    elif child.type == "aliased_import":
                        n = child.child_by_field_name("name")
                        a = child.child_by_field_name("alias")
                        if n and a:
                            self.import_mappings[filepath][self._txt(a)] = (self._txt(n), self._txt(n))
            elif node.type == "import_from_statement":
                mod_node = node.child_by_field_name("module_name")
                if not mod_node:
                    continue
                mod = self._txt(mod_node)
                # ``from foo import *`` — names are not listed; index the target module's
                # top-level functions so callees like ``handle_*`` still resolve.
                if any(c.type == "wildcard_import" for c in node.children):
                    mp = self._resolve_module_path(filepath, mod)
                    if mp:
                        self._parse_file(mp)
                        for fn in self.function_defs.get(mp, {}):
                            self.import_mappings[filepath][fn] = (mod, fn)
                    continue
                for child in node.children:
                    if child.type == "dotted_name" and child != mod_node:
                        name = self._txt(child)
                        self.import_mappings[filepath][name] = (mod, name)
                    elif child.type == "aliased_import":
                        n = child.child_by_field_name("name")
                        a = child.child_by_field_name("alias")
                        if n:
                            orig = self._txt(n)
                            alias = self._txt(a) if a else orig
                            self.import_mappings[filepath][alias] = (mod, orig)

    def _extract_function_defs(self, filepath, tree):
        self.function_defs[filepath] = {}
        self.function_locations[filepath] = []
        for node in self._traverse(tree.root_node):
            if node.type not in PY_FUNCTION_TYPES:
                continue
            nnode = node.child_by_field_name("name")
            if nnode:
                name = self._function_name(node)
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                self.function_defs[filepath][name] = (sl, el, node)
                self.function_locations[filepath].append((name, sl, el, node))

    def _extract_class_nodes(self, filepath: str, tree: Tree) -> None:
        self.class_nodes[filepath] = {
            self._function_name(node): node
            for node in self._traverse(tree.root_node)
            if node.type == "class_definition" and node.child_by_field_name("name")
        }

    _TYPING_NAMES = {
        "Optional", "Union", "List", "Dict", "Set", "Tuple", "Type", "Any", "None",
        "list", "dict", "set", "tuple", "type", "str", "int", "float", "bool", "bytes",
        "typing", "Annotated", "ClassVar", "Final",
    }

    def _py_type_name(self, text: str) -> Optional[str]:
        """First non-typing name in an annotation: ``Optional[Client]`` -> ``Client``."""
        for name in re.findall(r"[A-Za-z_][\w.]*", text.strip("'\"")):
            if name.rsplit(".", 1)[-1] not in self._TYPING_NAMES:
                return name
        return None

    def _py_enclosing(self, node: Node) -> tuple[Optional[Node], Optional[str]]:
        """(nearest enclosing function, qualified name of the class it belongs to)."""
        func = None
        parent = node.parent
        while parent:
            if parent.type in PY_FUNCTION_TYPES and func is None:
                func = parent
            elif parent.type == "class_definition":
                return func, self._function_name(parent)
            parent = parent.parent
        return func, None

    def _py_param_types(self, func: Node) -> dict[str, str]:
        types: dict[str, str] = {}
        for param in (func.child_by_field_name("parameters") or func).children:
            if param.type not in ("typed_parameter", "typed_default_parameter"):
                continue
            name = param.child_by_field_name("name") or next(
                (c for c in param.children if c.type == "identifier"), None
            )
            annotation = param.child_by_field_name("type")
            type_name = self._py_type_name(self._txt(annotation)) if annotation else None
            if name is not None and type_name:
                types[self._txt(name)] = type_name
        return types

    def _extract_instance_map(self, filepath: str, tree: Tree) -> None:
        """Index which class a variable / ``self`` field holds.

        Sources: ``x = Foo(...)``, ``x: Foo = ...``, ``self.x = Foo(...)``,
        ``self.x = x`` with a typed parameter ``x: Foo``, typed parameters, and
        annotated class attributes.
        """
        instances: dict[tuple[str, str], str] = {}
        fields: dict[str, dict[str, str]] = {}
        for node in self._traverse(tree.root_node):
            if node.type in ("function_definition", "async_function_definition"):
                scope = self._function_name(node)
                for name, type_name in self._py_param_types(node).items():
                    if name not in ("self", "cls"):
                        instances.setdefault((scope, name), type_name)
                continue
            if node.type != "assignment":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            annotation = node.child_by_field_name("type")
            if left is None:
                continue
            func, owner_class = self._py_enclosing(node)
            cls_text = None
            if right is not None and right.type == "await":
                right = next((c for c in right.children if c.is_named), None)
            if right is not None and right.type == "call":
                constructor = right.child_by_field_name("function")
                if constructor is not None and constructor.type in ("identifier", "attribute"):
                    cls_text = self._txt(constructor)
            if cls_text is None and annotation is not None:
                cls_text = self._py_type_name(self._txt(annotation))
            if cls_text is None and right is not None and right.type == "identifier" and func is not None:
                cls_text = self._py_param_types(func).get(self._txt(right))
            if not cls_text:
                continue
            if left.type == "attribute":
                obj = left.child_by_field_name("object")
                attr = left.child_by_field_name("attribute")
                if obj is not None and attr is not None and owner_class and self._txt(obj) in ("self", "cls"):
                    fields.setdefault(owner_class, {}).setdefault(self._txt(attr), cls_text)
            else:
                if left.type == "identifier" and func is None and owner_class:
                    # class attribute ``client: Client = ...``
                    fields.setdefault(owner_class, {}).setdefault(self._txt(left), cls_text)
                # identifiers, and other targets such as ``_cache[key] = Client()``
                # keyed by their source text
                scope = self._function_name(func) if func is not None else ""
                instances[(scope, self._txt(left))] = cls_text
        self.instance_map[filepath] = instances
        self.field_types[filepath] = fields

    # ── classes ──

    def _resolve_class(self, filepath: str, name: str, depth: int = 0) -> Optional[tuple[str, str]]:
        """Locate class *name* as seen from *filepath*: ``(file, qualified class name)``."""
        if depth > 4 or not name:
            return None
        self._parse_file(filepath)
        if name in self.class_nodes.get(filepath, {}):
            return (filepath, name)
        imports = self.import_mappings.get(filepath, {})
        if name in imports:
            mod, orig = imports[name]
            mp = self._resolve_module_path(filepath, mod)
            return self._resolve_class(mp, orig, depth + 1) if mp and orig != mod else None
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):  # ``mod.Foo`` / ``pkg.mod.Foo``
            prefix = ".".join(parts[:i])
            if prefix in imports:
                mod, orig = imports[prefix]
                # ``import pkg.mod`` / ``import pkg.mod as m`` -> mod; ``from pkg import mod`` -> pkg.mod
                if orig == mod:
                    candidates = [mod]
                else:
                    candidates = [mod + orig if mod.endswith(".") else f"{mod}.{orig}", mod]
                for module in candidates:
                    mp = self._resolve_module_path(filepath, module)
                    if mp:
                        found = self._resolve_class(mp, ".".join(parts[i:]), depth + 1)
                        if found:
                            return found
        return None

    def _class_bases(self, filepath: str, qualified: str) -> list[str]:
        node = self.class_nodes.get(filepath, {}).get(qualified)
        supers = node.child_by_field_name("superclasses") if node is not None else None
        if supers is None:
            return []
        return [self._txt(c) for c in supers.children if c.type in ("identifier", "attribute")]

    def _resolve_module_path(self, filepath, module_name):
        fdir = os.path.dirname(filepath)
        if module_name.startswith("."):
            dots = len(module_name) - len(module_name.lstrip("."))
            base = fdir
            for _ in range(dots - 1):
                base = os.path.dirname(base)
            rem = module_name[dots:]
            if rem:
                parts = rem.split(".")
                for cand in [os.path.join(base, *parts) + ".py",
                             os.path.join(base, *parts, "__init__.py")]:
                    if os.path.exists(cand):
                        return cand
            else:
                cand = os.path.join(base, "__init__.py")
                if os.path.exists(cand):
                    return cand
        parts = module_name.split(".")
        for base in self._python_search_roots(fdir):
            for cand in [os.path.join(base, *parts) + ".py",
                         os.path.join(base, *parts, "__init__.py")]:
                if os.path.exists(cand):
                    return cand
        return None

    def _python_search_roots(self, fdir: str) -> list[str]:
        """Directories that may act as ``sys.path`` roots for absolute imports.

        Besides the project root and the importing file's directory, covers the
        ``src/`` layout (``src/pkg/server.py`` doing ``from pkg.util import x``)
        and nested projects (monorepos) via every ancestor up to the project root.
        """
        roots = [self.project_root, fdir, os.path.join(self.project_root, "src")]
        current = fdir
        while current.startswith(self.project_root + os.sep):
            current = os.path.dirname(current)
            roots.append(current)
            roots.append(os.path.join(current, "src"))
        seen: set[str] = set()
        return [r for r in roots if not (r in seen or seen.add(r)) and os.path.isdir(r)]

    def _extract_callees(self, func_node):
        callees = []
        for node in self._traverse_function_body(func_node, PY_FUNCTION_TYPES):
            if node.type == "call":
                fe = node.child_by_field_name("function")
                if fe is not None and fe.type in ("identifier", "attribute"):
                    callees.append(self._txt(fe))
        return callees

    def _resolve_callee(self, filepath, callee_name, func_name=None):
        if callee_name in self.function_defs.get(filepath, {}):
            return (filepath, callee_name)

        owner = func_name.rsplit(".", 1)[0] if func_name and "." in func_name else ""
        scoped_candidates = []
        if func_name:
            scoped_candidates.append(f"{func_name}.{callee_name}")
        if owner:
            scoped_candidates.append(f"{owner}.{callee_name}")
        for scoped_name in scoped_candidates:
            if scoped_name in self.function_defs.get(filepath, {}):
                return (filepath, scoped_name)

        imports = self.import_mappings.get(filepath, {})
        parts = callee_name.split(".")

        if len(parts) == 1:
            if callee_name in imports:
                mod, orig = imports[callee_name]
                mp = self._resolve_module_path(filepath, mod)
                if mp:
                    self._parse_file(mp)
                    if orig in self.function_defs.get(mp, {}):
                        return (mp, orig)
            return self._constructor(filepath, callee_name)

        first = parts[0]
        method = parts[-1]
        if first in {"self", "cls"}:
            owner_class = self._owner_class(filepath, func_name)
            if len(parts) == 3 and owner_class:
                # ``self.client.query()`` with ``self.client = Client(...)``
                field_class = self.field_types.get(filepath, {}).get(owner_class, {}).get(parts[1])
                if field_class:
                    return self._class_member(filepath, field_class, method)
                return None
            scoped_method = f"{owner}.{method}" if owner else method
            if scoped_method in self.function_defs.get(filepath, {}):
                return (filepath, scoped_method)
            if len(parts) == 2 and owner_class:  # inherited from a project base class
                return self._class_member(filepath, owner_class, method)
            return None
        if first in imports:
            mod, _ = imports[first]
            mp = self._resolve_module_path(filepath, mod)
            if mp:
                self._parse_file(mp)
                if method in self.function_defs.get(mp, {}):
                    return (mp, method)
        return (
            self._resolve_var_method(filepath, callee_name, func_name)
            or self._constructor(filepath, callee_name)
        )

    def _resolve_var_method(self, filepath, callee_name, func_name=None):
        parts = callee_name.split(".")
        if len(parts) != 2:
            return None
        var, method = parts
        source = filepath
        instances = self.instance_map.get(filepath, {})
        scopes = (func_name or "").split(".")
        # current function, then enclosing functions / classes, then module level
        cls = next(
            (instances[key] for i in range(len(scopes), -1, -1)
             if (key := (".".join(scopes[:i]), var)) in instances),
            None,
        )
        if not cls:
            # Assigned in another function (e.g. ``global client; client = Client()``
            # in an init function): accept it when the file binds the name to one class.
            cls = self._unique_instance_class(instances, var)
        if not cls:
            imports = self.import_mappings.get(filepath, {})
            if var in imports:  # ``from .db import db_context`` where db.py does ``db_context = DB()``
                mod, orig = imports[var]
                mp = self._resolve_module_path(filepath, mod)
                if mp:
                    self._parse_file(mp)
                    cls = self._unique_instance_class(self.instance_map.get(mp, {}), orig)
                    source = mp
        if not cls:
            return None
        return self._class_member(source, cls, method)



# ─────────────────────────────────────────────────────────
# TypeScript / JavaScript slicer
# ─────────────────────────────────────────────────────────

class TSSlicer(_BaseSlicer):
    FUNCTION_TYPES = JS_FUNCTION_TYPES
    CONSTRUCTOR = "constructor"

    def __init__(self, project_root: str):
        super().__init__(project_root, Parser(TS_LANG))
        self.js_parser = Parser(JS_LANG)
        self.ts_parser = self.parser
        self.tsx_parser = Parser(TSX_LANG)
        self.import_map: dict[str, dict[str, tuple[str, str]]] = {}
        # (scope function, variable) -> class expression (``const c = new Client()`` / ``c: Client``)
        self.instance_map: dict[str, dict[tuple[str, str], str]] = {}
        self.reexport_map: dict[str, list[tuple[str, str, Optional[str], Optional[str]]]] = {}
        # name (simple, qualified or "default") -> qualified class name
        self.class_defs: dict[str, dict[str, str]] = {}
        # qualified class name -> field -> class expression (``this.x = new Foo()``)
        self.field_types: dict[str, dict[str, dict[str, str]]] = {}

    def _parser_for_file(self, filepath: str) -> Parser:
        ext = Path(filepath).suffix.lower()
        if ext == ".tsx":
            return self.tsx_parser
        if ext in {".ts", ".mts", ".cts"}:
            return self.ts_parser
        return self.js_parser

    def _top_level_assignments(self, filepath: str):
        tree, _ = self.file_cache[filepath]
        for stmt in tree.root_node.children:
            decls = [stmt]
            if stmt.type == "export_statement":
                decls = stmt.named_children
            for decl in decls:
                if decl.type not in {"lexical_declaration", "variable_declaration"}:
                    continue
                for declarator in decl.named_children:
                    name = declarator.child_by_field_name("name") \
                        if declarator.type == "variable_declarator" else None
                    if name is not None and name.type == "identifier":
                        yield self._txt(name), stmt

    def _function_name(self, node: Node) -> str:
        name_node = node.child_by_field_name("name")
        base = self._txt(name_node) if name_node else f"[inline:{node.start_point[0] + 1}]"
        return self._qualified_name(node, base)

    def _qualified_name(self, node: Node, base: str) -> str:
        """Prefix *base* with enclosing class / function names.

        Anonymous functions are qualified too (``MCPRestServer.[inline:428]``) so
        that ``this.method()`` inside a handler arrow resolves against its class.
        """
        parts = [base]
        parent = node.parent
        while parent:
            if parent.type in {"class_declaration", "class"}:
                owner = parent.child_by_field_name("name")
                if owner:
                    parts.append(self._txt(owner))
            elif parent.type == "function_declaration":
                owner = parent.child_by_field_name("name")
                if owner:
                    parts.append(self._txt(owner))
            parent = parent.parent
        return ".".join(reversed(parts))

    def _index_file(self, filepath, tree):
        self._extract_imports(filepath, tree)
        self._extract_reexports(filepath, tree)
        self._extract_class_defs(filepath, tree)
        self._extract_function_defs(filepath, tree)
        self._extract_instance_map(filepath, tree)

    # ── imports ──

    def _extract_imports(self, filepath, tree):
        self.import_map[filepath] = {}
        for node in tree.root_node.children:
            if node.type in {"lexical_declaration", "variable_declaration"}:
                self._extract_require_imports(filepath, node)
                continue
            if node.type != "import_statement":
                continue
            src_node = node.child_by_field_name("source")
            if not src_node:
                continue
            src_str = self._txt(src_node).strip("'\"")
            if not src_str.startswith("."):
                continue
            resolved = self._resolve_import_path(os.path.dirname(filepath), src_str)
            if not resolved:
                continue
            for child in node.children:
                if child.type == "import_clause":
                    for cc in child.children:
                        if cc.type == "named_imports":
                            for spec in cc.children:
                                if spec.type == "import_specifier":
                                    n = spec.child_by_field_name("name")
                                    a = spec.child_by_field_name("alias")
                                    if n:
                                        orig = self._txt(n)
                                        local = self._txt(a) if a else orig
                                        self.import_map[filepath][local] = (resolved, orig)
                        elif cc.type == "namespace_import":
                            alias = next(
                                (c for c in reversed(cc.children) if c.type == "identifier"),
                                None,
                            )
                            if alias:
                                self.import_map[filepath][self._txt(alias)] = (resolved, "*")
                        elif cc.type == "identifier":
                            self.import_map[filepath][self._txt(cc)] = (resolved, "default")

    def _extract_require_imports(self, filepath: str, declaration: Node) -> None:
        """Index local CommonJS require imports."""
        for declarator in declaration.children:
            if declarator.type != "variable_declarator":
                continue
            name = declarator.child_by_field_name("name")
            value = declarator.child_by_field_name("value")
            if not name or not value or value.type != "call_expression":
                continue
            callee = value.child_by_field_name("function")
            arguments = value.child_by_field_name("arguments")
            if not callee or self._txt(callee) != "require" or not arguments:
                continue
            source_node = next((c for c in arguments.children if c.type == "string"), None)
            if not source_node:
                continue
            source_path = self._txt(source_node).strip("'\"")
            if not source_path.startswith("."):
                continue
            resolved = self._resolve_import_path(os.path.dirname(filepath), source_path)
            if not resolved:
                continue
            if name.type == "identifier":
                self.import_map[filepath][self._txt(name)] = (resolved, "*")
                continue
            if name.type != "object_pattern":
                continue
            for item in name.children:
                if item.type == "shorthand_property_identifier_pattern":
                    imported = self._txt(item)
                    self.import_map[filepath][imported] = (resolved, imported)
                elif item.type == "pair_pattern":
                    key = item.child_by_field_name("key")
                    local = item.child_by_field_name("value")
                    if key and local:
                        self.import_map[filepath][self._txt(local)] = (
                            resolved,
                            self._txt(key),
                        )

    def _resolve_import_path(self, base_dir, import_path):
        # TS MCP repos frequently import "./foo.mjs" while source lives in "./foo.mts".
        # Normalize JS module suffixes and probe TS/JS variants in a predictable order.
        normalized = import_path
        for suffix in (".mjs", ".cjs", ".js"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        resolved = os.path.normpath(os.path.join(base_dir, normalized))
        candidates = [
            resolved,
            resolved + ".mts",
            resolved + ".cts",
            resolved + ".ts",
            resolved + ".tsx",
            resolved + ".mjs",
            resolved + ".cjs",
            resolved + ".js",
            resolved + ".jsx",
            os.path.join(resolved, "index.mts"),
            os.path.join(resolved, "index.cts"),
            os.path.join(resolved, "index.ts"),
            os.path.join(resolved, "index.tsx"),
            os.path.join(resolved, "index.mjs"),
            os.path.join(resolved, "index.cjs"),
            os.path.join(resolved, "index.js"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    # ── re-exports / barrel files ──

    def _extract_reexports(self, filepath, tree):
        self.reexport_map[filepath] = []
        for node in tree.root_node.children:
            if node.type != "export_statement":
                continue
            src_node = node.child_by_field_name("source")
            src_str = self._txt(src_node).strip("'\"") if src_node else ""
            if not src_str.startswith("."):
                continue
            resolved = self._resolve_import_path(os.path.dirname(filepath), src_str)
            if not resolved:
                continue

            if any(child.type == "*" for child in node.children):
                self.reexport_map[filepath].append(("star", resolved, None, None))
                continue

            for child in node.children:
                if child.type != "export_clause":
                    continue
                for spec in child.children:
                    if spec.type != "export_specifier":
                        continue
                    ids = [
                        c for c in spec.children
                        if c.type in ("identifier", "property_identifier")
                    ]
                    if not ids:
                        continue
                    imported = self._txt(ids[0])
                    exported = self._txt(ids[-1])
                    self.reexport_map[filepath].append(
                        ("named", resolved, exported, imported)
                    )

    def _resolve_exported_symbol(
        self,
        filepath: str,
        symbol: str,
        seen: Optional[set[tuple[str, str]]] = None,
    ) -> Optional[tuple[str, str]]:
        if seen is None:
            seen = set()
        key = (filepath, symbol)
        if key in seen:
            return None
        seen.add(key)

        self._parse_file(filepath)
        if symbol in self.function_defs.get(filepath, {}):
            return (filepath, self._canonical_function_name(filepath, symbol))

        for kind, target, exported, imported in self.reexport_map.get(filepath, []):
            if kind == "named" and exported == symbol and imported:
                resolved = self._resolve_exported_symbol(target, imported, seen)
            elif kind == "star":
                resolved = self._resolve_exported_symbol(target, symbol, seen)
            else:
                continue
            if resolved:
                return resolved
        return None

    def _canonical_function_name(self, filepath: str, name: str) -> str:
        """Return the concrete name when an export alias points at a function."""
        info = self.function_defs.get(filepath, {}).get(name)
        if not info:
            return name
        for candidate, candidate_info in self.function_defs.get(filepath, {}).items():
            if candidate != "default" and candidate_info[2] == info[2]:
                return candidate
        return name

    def _resolve_identifier_function(self, filepath: str, name: str) -> Optional[tuple[str, str]]:
        """Function *name* as seen from *filepath*: local, imported, or re-exported."""
        self._parse_file(filepath)
        imported = self.import_map.get(filepath, {}).get(name)
        if imported and name not in self.function_defs.get(filepath, {}):
            resolved = self._resolve_exported_symbol(*imported)
            if resolved:
                return resolved
        return self._resolve_exported_symbol(filepath, name)

    # ── instance map (new ClassName) ──

    _TS_CLASS_TYPES = {"class_declaration", "class", "abstract_class_declaration"}
    _TS_NON_CLASS_TYPES = {
        "Promise", "Array", "Record", "Partial", "Readonly", "Required", "Map", "Set",
        "string", "number", "boolean", "any", "unknown", "undefined", "null", "void",
        "object", "never", "ReturnType", "InstanceType", "Awaited",
    }

    def _ts_type_name(self, text: str) -> Optional[str]:
        """First class-like name in a type annotation: ``: Promise<Client> | null`` -> ``Client``."""
        for name in re.findall(r"[A-Za-z_$][\w$.]*", text):
            if name.rsplit(".", 1)[-1] not in self._TS_NON_CLASS_TYPES:
                return name
        return None

    def _ts_value_class(self, value: Optional[Node]) -> Optional[str]:
        """Class expression for ``new Foo(...)`` (also under ``await`` / parentheses / casts)."""
        while value is not None and value.type in (
            "await_expression", "parenthesized_expression", "as_expression", "non_null_expression"
        ):
            value = next((c for c in value.children if c.is_named), None)
        if value is not None and value.type == "new_expression":
            ctor = value.child_by_field_name("constructor")
            if ctor is not None and ctor.type in ("identifier", "member_expression"):
                return self._txt(ctor)
        return None

    def _ts_enclosing_class(self, node: Node) -> Optional[str]:
        parent = node.parent
        while parent:
            if parent.type in self._TS_CLASS_TYPES:
                name = parent.child_by_field_name("name")
                return self._qualified_name(parent, self._txt(name)) if name else None
            parent = parent.parent
        return None

    def _ts_param_types(self, func: Optional[Node]) -> dict[str, str]:
        types: dict[str, str] = {}
        params = func.child_by_field_name("parameters") if func is not None else None
        for param in params.children if params is not None else []:
            if param.type not in ("required_parameter", "optional_parameter"):
                continue
            pattern = param.child_by_field_name("pattern")
            annotation = param.child_by_field_name("type")
            type_name = self._ts_type_name(self._txt(annotation)) if annotation else None
            if pattern is not None and pattern.type == "identifier" and type_name:
                types[self._txt(pattern)] = type_name
        return types

    def _extract_class_defs(self, filepath: str, tree: Tree) -> None:
        names: dict[str, str] = {}
        nodes: dict[str, Node] = {}
        for node in self._traverse(tree.root_node):
            if node.type not in self._TS_CLASS_TYPES:
                continue
            name = node.child_by_field_name("name")
            if name is None:
                continue
            qualified = self._qualified_name(node, self._txt(name))
            nodes[qualified] = node
            names.setdefault(self._txt(name), qualified)
            names[qualified] = qualified
            if node.parent is not None and node.parent.type == "export_statement" and \
                    self._txt(node.parent).lstrip().startswith("export default"):
                names["default"] = qualified
        for node in tree.root_node.children:  # ``class Foo {}; export default Foo;``
            if node.type == "export_statement" and self._txt(node).lstrip().startswith("export default"):
                value = node.child_by_field_name("value")
                if value is not None and value.type == "identifier" and self._txt(value) in names:
                    names.setdefault("default", names[self._txt(value)])
        self.class_defs[filepath] = names
        self.class_nodes[filepath] = nodes

    def _extract_instance_map(self, filepath, tree):
        """Index which class a variable / ``this`` field holds.

        Sources: ``const c = new Foo()``, ``c = new Foo()``, ``const c: Foo``,
        ``this.x = new Foo()``, ``this.x = x`` with a typed parameter, class fields
        (``private x = new Foo()`` / ``x: Foo``) and constructor parameter
        properties (``constructor(private x: Foo)``).
        """
        instances: dict[tuple[str, str], str] = {}
        fields: dict[str, dict[str, str]] = {}

        def scope_of(node: Node) -> str:
            parent = node.parent
            while parent is not None:
                if parent.type in JS_FUNCTION_TYPES:
                    name = self._function_name_at(filepath, parent)
                    if name:
                        return name
                parent = parent.parent
            return ""

        for node in self._traverse(tree.root_node):
            if node.type == "variable_declarator":
                name = node.child_by_field_name("name")
                annotation = node.child_by_field_name("type")
                cls = self._ts_value_class(node.child_by_field_name("value")) or (
                    self._ts_type_name(self._txt(annotation)) if annotation else None
                )
                if name is not None and name.type == "identifier" and cls:
                    instances[(scope_of(node), self._txt(name))] = cls
            elif node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                cls = self._ts_value_class(right)
                if left is None:
                    continue
                if left.type == "member_expression":
                    obj = left.child_by_field_name("object")
                    prop = left.child_by_field_name("property")
                    owner = self._ts_enclosing_class(node)
                    if obj is None or prop is None or obj.type != "this" or not owner:
                        continue
                    if cls is None and right is not None and right.type == "identifier":
                        func = node.parent
                        while func is not None and func.type not in JS_FUNCTION_TYPES:
                            func = func.parent
                        cls = self._ts_param_types(func).get(self._txt(right))
                    if cls:
                        fields.setdefault(owner, {}).setdefault(self._txt(prop), cls)
                elif left.type == "identifier" and cls:
                    instances[(scope_of(node), self._txt(left))] = cls
            elif node.type in ("public_field_definition", "field_definition"):
                name = node.child_by_field_name("name") or node.child_by_field_name("property")
                annotation = node.child_by_field_name("type")
                owner = self._ts_enclosing_class(node)
                cls = self._ts_value_class(node.child_by_field_name("value")) or (
                    self._ts_type_name(self._txt(annotation)) if annotation else None
                )
                if name is not None and owner and cls:
                    fields.setdefault(owner, {}).setdefault(self._txt(name), cls)
            elif node.type in ("required_parameter", "optional_parameter"):
                if not any(c.type in ("accessibility_modifier", "readonly") for c in node.children):
                    continue  # only constructor parameter properties become fields
                pattern = node.child_by_field_name("pattern")
                annotation = node.child_by_field_name("type")
                owner = self._ts_enclosing_class(node)
                cls = self._ts_type_name(self._txt(annotation)) if annotation else None
                if pattern is not None and owner and cls:
                    fields.setdefault(owner, {}).setdefault(self._txt(pattern), cls)
        self.instance_map[filepath] = instances
        self.field_types[filepath] = fields

    # ── classes ──

    def _resolve_class(
        self, filepath: str, name: str, seen: Optional[set[tuple[str, str]]] = None
    ) -> Optional[tuple[str, str]]:
        """Locate class *name* as seen from *filepath*: ``(file, qualified class name)``."""
        seen = set() if seen is None else seen
        if (filepath, name) in seen or not name:
            return None
        seen.add((filepath, name))
        self._parse_file(filepath)
        classes = self.class_defs.get(filepath, {})
        if name in classes:
            return (filepath, classes[name])
        imports = self.import_map.get(filepath, {})
        if "." in name:  # ``ns.Foo`` with ``import * as ns``
            ns, rest = name.split(".", 1)
            if ns in imports and imports[ns][1] == "*":
                return self._resolve_class(imports[ns][0], rest, seen)
            return None
        if name in imports:
            target, orig = imports[name]
            if orig != "*":
                return self._resolve_class(target, orig, seen)
        for kind, target, exported, imported in self.reexport_map.get(filepath, []):
            if kind == "named" and exported == name and imported:
                found = self._resolve_class(target, imported, seen)
            elif kind == "star":
                found = self._resolve_class(target, name, seen)
            else:
                continue
            if found:
                return found
        return None

    def _class_bases(self, filepath: str, qualified: str) -> list[str]:
        node = self.class_nodes.get(filepath, {}).get(qualified)
        heritage = next((c for c in node.children if c.type == "class_heritage"), None) if node else None
        if heritage is None:
            return []
        bases = []
        for item in self._traverse(heritage):
            if item.type == "extends_clause":
                value = item.child_by_field_name("value")
                if value is not None:
                    bases.append(self._txt(value))
        if not bases:  # JavaScript: class_heritage -> ``extends`` expression
            value = next((c for c in heritage.children if c.is_named), None)
            if value is not None and value.type in ("identifier", "member_expression"):
                bases.append(self._txt(value))
        return bases

    def _ts_instance_class(self, filepath: str, var: str, func_name: Optional[str]) -> Optional[str]:
        """Class held by *var*: current function, enclosing functions, module, else unique in file.

        Enclosing functions are found by walking the syntax tree upward from the
        function node itself; re-looking them up by name could jump to a different
        function with the same name and loop forever.
        """
        instances = self.instance_map.get(filepath, {})
        if func_name:
            if (func_name, var) in instances:
                return instances[(func_name, var)]
            info = self.function_defs.get(filepath, {}).get(func_name)
            parent = info[2].parent if info else None
            while parent is not None:
                if parent.type in JS_FUNCTION_TYPES:
                    name = self._function_name_at(filepath, parent)
                    if name and (name, var) in instances:
                        return instances[(name, var)]
                parent = parent.parent
        if ("", var) in instances:
            return instances[("", var)]
        return self._unique_instance_class(instances, var)

    # ── function defs ──

    def _extract_function_defs(self, filepath, tree):
        self.function_defs[filepath] = {}
        self.function_locations[filepath] = []

        for node in self._traverse(tree.root_node):
            names: list[str] = []
            function_node = node

            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    names.append(self._function_name(node))
                elif node.parent and node.parent.type == "export_statement":
                    names.append("default")
                if node.parent and node.parent.type == "export_statement":
                    export_text = self._txt(node.parent).lstrip()
                    if export_text.startswith("export default"):
                        names.append("default")

            elif node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if name_node and value and value.type in {
                    "arrow_function", "function_expression", "function"
                }:
                    names.append(self._txt(name_node))
                    function_node = value

            elif node.type == "method_definition":
                names.append(self._function_name(node))

            elif node.type == "pair":
                key = node.child_by_field_name("key")
                value = node.child_by_field_name("value")
                if key and value and value.type in {
                    "arrow_function", "function_expression", "function"
                }:
                    key_name = self._txt(key).strip("'\"")
                    names.append(f"{key_name}:{value.start_point[0] + 1}")
                    function_node = value

            elif node.type in {"arrow_function", "function_expression"}:
                parent = node.parent
                if parent and parent.type not in {
                    "variable_declarator", "assignment_expression", "method_definition", "pair"
                }:
                    names.append(
                        self._qualified_name(node, f"[inline:{node.start_point[0] + 1}]")
                    )

            if not names:
                continue
            info = (
                function_node.start_point[0] + 1,
                function_node.end_point[0] + 1,
                function_node,
            )
            for name in names:
                self.function_defs[filepath][name] = info
            location_name = next((name for name in names if name != "default"), names[0])
            self.function_locations[filepath].append((location_name, *info))

        # ``const handler = async () => {...}; export default handler;``
        for node in tree.root_node.children:
            if node.type != "export_statement" or not self._txt(node).lstrip().startswith("export default"):
                continue
            value = node.child_by_field_name("value")
            if value is not None and value.type == "identifier":
                info = self.function_defs[filepath].get(self._txt(value))
                if info:
                    self.function_defs[filepath].setdefault("default", info)

    # ── callees ──

    def _extract_callees(self, func_node):
        callees = []
        for node in self._traverse_function_body(func_node, JS_FUNCTION_TYPES):
            if node.type == "new_expression":
                ctor = node.child_by_field_name("constructor")
                if ctor is not None and ctor.type in ("identifier", "member_expression"):
                    callees.append(f"new {self._txt(ctor)}")
                continue
            if node.type != "call_expression":
                continue
            fe = node.child_by_field_name("function")
            if fe is None:
                continue
            if fe.type == "identifier":
                callees.append(self._txt(fe))
            elif fe.type == "member_expression":
                obj = fe.child_by_field_name("object")
                prop = fe.child_by_field_name("property")
                if obj and prop:
                    callees.append(f"{self._txt(obj)}.{self._txt(prop)}")
        return callees

    def _resolve_callee(self, filepath, callee_name, func_name=None):
        if callee_name.startswith("new "):
            # ``new Foo(...)`` on a project class -> ``Foo.constructor`` (exceptions skipped)
            return self._constructor(filepath, callee_name[4:])
        if "." in callee_name:
            obj, method = callee_name.split(".", 1)
            owner = func_name.rsplit(".", 1)[0] if func_name and "." in func_name else ""
            if obj == "this" and "." in method:
                # ``this.client.query()`` with ``this.client = new Client()`` or a
                # ``constructor(private client: Client)`` parameter property
                field, member = method.split(".", 1)
                owner_class = self._owner_class(filepath, func_name)
                field_class = self.field_types.get(filepath, {}).get(owner_class or "", {}).get(field)
                if field_class and "." not in member:
                    return self._class_member(filepath, field_class, member)
                return None
            instance_class = self._ts_instance_class(filepath, obj, func_name)
            if instance_class:
                found = self._class_member(filepath, instance_class, method)
                if found:
                    return found
            imp = self.import_map.get(filepath, {})
            if obj in imp:
                cf, imported = imp[obj]
                self._parse_file(cf)
                if imported == "*":
                    return self._resolve_exported_symbol(cf, method)
                qualified = f"{imported}.{method}"
                if qualified in self.function_defs.get(cf, {}):
                    return (cf, qualified)
                if method in self.function_defs.get(cf, {}):
                    return (cf, self._canonical_function_name(cf, method))
                # imported instance: ``export const client = new Client()``
                instance_class = self._ts_instance_class(cf, imported, None)
                if instance_class:
                    return self._class_member(cf, instance_class, method)
            if obj == "this":
                qualified = f"{owner}.{method}" if owner else method
                if qualified in self.function_defs.get(filepath, {}):
                    return (filepath, qualified)
                owner_class = self._owner_class(filepath, func_name)
                if owner_class:  # inherited from a project base class
                    return self._class_member(filepath, owner_class, method)
            return None

        direct = self._resolve_identifier_function(filepath, callee_name)
        if direct:
            return direct
        owner = func_name.rsplit(".", 1)[0] if func_name and "." in func_name else ""
        for prefix in filter(None, (func_name, owner)):
            scoped = f"{prefix}.{callee_name}"
            if scoped in self.function_defs.get(filepath, {}):
                return (filepath, scoped)
        return None


# ─────────────────────────────────────────────────────────
# Slice generation
# ─────────────────────────────────────────────────────────

def _make_slicer(project_root: str, ext: str) -> _BaseSlicer:
    if ext == ".py":
        return PythonSlicer(project_root)
    if ext in SUPPORTED_EXTENSIONS:
        return TSSlicer(project_root)
    raise ValueError(f"Unsupported source extension: {ext or '<none>'}")


def _comment(ext: str) -> str:
    return "#" if ext == ".py" else "//"


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^\w\-.]", "_", value) or "_"


def _tool_directory_names(tools: list[dict]) -> dict[str, str]:
    """Return readable, case-insensitive-filesystem-safe tool directories."""
    groups: dict[str, set[str]] = {}
    for tool in tools:
        name = str(tool.get("tool_name") or "")
        groups.setdefault(_safe_path_component(name).casefold(), set()).add(name)

    directories: dict[str, str] = {}
    for names in groups.values():
        for name in names:
            component = _safe_path_component(name)
            if len(names) > 1:
                digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
                component = f"{component}__{digest}"
            directories[name] = component
    return directories


def _set_slice_result(tool: dict, path: Optional[str]) -> None:
    tool["slice_generated"] = path is not None
    tool["slice_path"] = path


def _select_within_budget(
    call_graph: dict[str, dict], order: list[str], max_lines: Optional[int]
) -> tuple[set[str], list[str]]:
    """Pick functions to keep under *max_lines*: entry first, then shallowest calls.

    Each function costs its line count plus a 4-line header. The entry function
    is always kept. Returns ``(kept ids, omitted ids in call-depth order)``.
    """
    if not max_lines or not order:
        return set(order), []
    used = 2  # "Tool:" header
    kept: set[str] = set()
    omitted: list[str] = []
    by_depth = sorted(order, key=lambda f: (call_graph[f].get("depth", 0), int(f.split("_")[1])))
    for fid in by_depth:
        cost = call_graph[fid]["content"].count("\n") + 1 + 4
        if fid == order[0] or used + cost <= max_lines:
            kept.add(fid)
            used += cost
        else:
            omitted.append(fid)
    return kept, omitted


def generate_slice_text(
    call_graph: dict[str, dict[str, str]],
    ext: str,
    tool_name: str = "",
    max_lines: Optional[int] = None,
    info: Optional[dict] = None,
) -> str:
    """Render a slice. With *max_lines*, deep callees beyond the budget are omitted
    and listed at the end; *info* (if given) receives ``truncated`` / ``omitted_functions``."""
    c = _comment(ext)
    parts: list[str] = []
    if tool_name:
        parts.append(f"{c} Tool: {tool_name}")
        parts.append("")

    order = sorted(call_graph, key=lambda x: int(x.split("_")[1]))
    kept, omitted = _select_within_budget(call_graph, order, max_lines)
    for fid in order:
        if fid not in kept:
            continue
        fi = call_graph[fid]
        parts.append(f"{c} {'='*60}")
        parts.append(f"{c} {fi['func_name']}")
        parts.append(f"{c} {'='*60}")
        parts.append(fi["content"])
        parts.append("")

    if omitted:
        parts.append(f"{c} {'='*60}")
        parts.append(
            f"{c} [slice truncated at ~{max_lines} lines: {len(omitted)} deeper "
            f"function(s) omitted; inspect them in the repository if needed]"
        )
        for fid in omitted[:50]:
            parts.append(f"{c}   - {call_graph[fid]['func_name']}")
        if len(omitted) > 50:
            parts.append(f"{c}   ... and {len(omitted) - 50} more")
        parts.append("")
    if info is not None:
        info["truncated"] = bool(omitted)
        info["omitted_functions"] = len(omitted)
        info["total_functions"] = len(order)
    return "\n".join(parts)


def generate_slice_for_tool(
    project_root: str,
    entry_loc: str,
    tool_name: str,
    server_name: str,
    output_base: Path,
    slicer_cache: Optional[dict[str, _BaseSlicer]] = None,
    tool_directory: str = "",
    entry_block: Optional[dict] = None,
    entry_handler: Optional[str | list[str]] = None,
    max_lines: Optional[int] = None,
    info: Optional[dict] = None,
) -> Optional[str]:
    """
    Generate a code slice for a single tool.

    Returns the output path on success, None on failure.
    """
    try:
        filepath, line_str = entry_loc.rsplit(":", 1)
        line = int(line_str)
        filepath = str(Path(filepath).expanduser().resolve())
        project_path = Path(project_root).expanduser().resolve()
        if os.path.commonpath((str(project_path), filepath)) != str(project_path):
            raise ValueError(f"Entry is outside the server source tree: {filepath}")

        ext = Path(filepath).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported source extension: {ext or '<none>'}")
        cache_key = "python" if ext == ".py" else "typescript-javascript"
        slicer = slicer_cache.get(cache_key) if slicer_cache is not None else None
        if slicer is None:
            slicer = _make_slicer(str(project_path), ext)
            if slicer_cache is not None:
                slicer_cache[cache_key] = slicer
        if entry_block:
            call_graph = slicer.analyze_block(
                filepath,
                int(entry_block["start_byte"]),
                int(entry_block["end_byte"]),
                tool_name,
                expected_start_line=entry_block.get("start_line"),
            )
        else:
            call_graph = slicer.analyze(filepath, line, handler=entry_handler)
        slice_text = generate_slice_text(call_graph, ext, tool_name, max_lines=max_lines, info=info)

        safe_tool = tool_directory or _safe_path_component(tool_name)
        safe_server = _safe_path_component(server_name)
        out_dir = output_base / safe_server / safe_tool
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "slice"
        out_file.write_text(slice_text, encoding="utf-8")
        return str(out_file)
    except Exception as exc:
        logger.warning("  Slice failed for %s: %s", tool_name, exc)
        return None


def generate_slices(
    extract_json: Path,
    output_base: Path,
    single_server_name: str = "",
    single_tool_name: str = "",
    overwrite: bool = False,
    max_lines: Optional[int] = MAX_SLICE_LINES,
    timeout_sec: Optional[float] = SLICE_TIMEOUT_SEC,
) -> dict[str, int]:
    """
    Read tool_metadata.json, generate slices for every matched tool,
    mark each tool with slice_generated, and save back.

    When ``single_server_name``/``single_tool_name`` are set, only the matched
    tool target is considered.
    """
    data = json.loads(extract_json.read_text(encoding="utf-8"))
    results = data.get("results", [])

    total_tools = 0
    sliced = 0
    failed = 0
    skipped = 0
    truncated = 0
    timed_out = 0

    for server in results:
        server_name = server["name"]
        if single_server_name and server_name != single_server_name:
            continue
        project_root = server.get("local_repo_dir")
        candidate = Path(project_root) if project_root else None
        selected_tools = [
            tool
            for tool in server.get("tools", [])
            if not single_tool_name or tool.get("tool_name") == single_tool_name
        ]
        tool_directories = _tool_directory_names(selected_tools)
        if candidate is None or not candidate.is_dir():
            logger.warning("Local repo dir not found for %s, skipping", server_name)
            for tool in selected_tools:
                total_tools += 1
                failed += 1
                _set_slice_result(tool, None)
            continue
        project_root = str(candidate)

        seen_tool_names: set[str] = set()
        slicer_cache: dict[str, _BaseSlicer] = {}
        for tool in selected_tools:
            tname = str(tool.get("tool_name") or "")
            total_tools += 1
            normalized_tname = tname.strip()
            if normalized_tname in seen_tool_names:
                logger.info("  Skipping duplicate tool name: %s / %s", server_name, tname)
                _set_slice_result(tool, None)
                tool["duplicate_tool_skipped"] = True
                skipped += 1
                continue
            if normalized_tname:
                seen_tool_names.add(normalized_tname)
            entry = tool.get("entry", "")
            tdesc = str(tool.get("tool_description") or "")
            if not normalized_tname or not tdesc.strip():
                logger.info(
                    "  Skipping tool with empty name or description: %s / %s",
                    server_name,
                    tname,
                )
                _set_slice_result(tool, None)
                failed += 1
                continue
            existing_slice = tool.get("slice_path")
            expected_slice = (
                output_base
                / _safe_path_component(server_name)
                / tool_directories[tname]
                / "slice"
            )
            if (
                not overwrite
                and tool.get("slice_generated") is True
                and existing_slice
                and Path(existing_slice).is_file()
                and Path(existing_slice).resolve() == expected_slice.resolve()
            ):
                logger.info("  Skipping existing slice: %s / %s", server_name, tname)
                skipped += 1
                continue
            if (
                not overwrite
                and tool.get("timeout") is True
                and tool.get("timeout_stage") == "slice"
            ):
                logger.info("  Skipping tool that timed out while slicing: %s / %s", server_name, tname)
                skipped += 1
                continue
            tool.pop("duplicate_tool_skipped", None)
            logger.info("  Slicing: %s / %s", server_name, tname)
            slice_info: dict = {}
            try:
                with _time_limit(timeout_sec):
                    out = generate_slice_for_tool(
                        project_root,
                        entry,
                        tname,
                        server_name,
                        output_base,
                        slicer_cache=slicer_cache,
                        tool_directory=tool_directories[tname],
                        entry_block=tool.get("entry_block"),
                        entry_handler=tool.get("entry_handler"),
                        max_lines=max_lines,
                        info=slice_info,
                    )
            except SliceTimeout:
                # Partially built indexes may be inconsistent: rebuild them for later tools.
                slicer_cache.clear()
                timed_out += 1
                _set_slice_result(tool, None)
                tool["timeout"] = True
                tool["timeout_stage"] = "slice"
                logger.warning(
                    "    slicing timed out after %ss; later stages will skip this tool", timeout_sec
                )
                continue
            if tool.get("timeout_stage") == "slice":
                tool.pop("timeout", None)
                tool.pop("timeout_stage", None)
            if out:
                sliced += 1
                _set_slice_result(tool, out)
                if slice_info.get("truncated"):
                    tool["slice_truncated"] = True
                    truncated += 1
                    logger.info(
                        "    → %s (truncated: %d of %d functions omitted)",
                        out, slice_info["omitted_functions"], slice_info["total_functions"],
                    )
                else:
                    tool.pop("slice_truncated", None)
                    logger.info("    → %s", out)
            else:
                failed += 1
                _set_slice_result(tool, None)

    extract_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "total": total_tools, "sliced": sliced, "failed": failed,
        "skipped": skipped, "truncated": truncated, "timed_out": timed_out,
    }
