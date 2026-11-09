Describe the actual behavior of an MCP (Model Context Protocol) tool's implementation in a stable format for later comparison with the author's description.

Use only the supplied evidence. Report behavior without inferring intent, quality, safety, or maliciousness.

## Inputs

**Tool Name:** {TOOL_NAME}

**Code Slice:** {CODE_SLICE}

**Server Memory:** {SERVER_MEMORY}

## Critical Operations

Mark a step `Critical: yes` when it performs an action a user would need to know about before trusting the tool: one that crosses a process, host, or trust boundary, touches persistent or sensitive state, or consumes significant resources. This includes:

- Filesystem: reading, writing, creating, deleting, or changing permissions of files or directories
- Network: requests to APIs, webhooks, sockets, or local services, including what data is sent
- Execution: spawning processes or shell commands, or evaluating dynamic code
- Sensitive data: transmitting or storing credentials, tokens, keys, cookies, or personal data
- Persistent or shared state: accessing databases, caches, browsers, or the clipboard; or changing environment variables, OS settings, or scheduled tasks
- Resources: unbounded loops, large downloads or allocations, or spawning many processes or tasks

Not critical: bounded computation, validation, parsing, formatting, in-memory transformation, routine control flow, diagnostic logging that records no sensitive user data (e.g., credentials, tokens, keys, cookies, or personal data), and concurrency primitives by themselves. Reading static files or configuration bundled with the server's own code at fixed paths is not critical; sending their contents across a boundary is.

## Repository Inspection

Use the available repository tools only when both conditions hold:

1. An unresolved symbol or value participates in a critical operation.
2. Its definition is absent from both the code slice and server memory.

Look up a definition with `find_definition` first; use `grep_repo` or `read_file` only when it finds nothing. Do not inspect non-critical helpers, standard-library behavior, or well-known third-party behavior. Treat server-memory entries as confirmed definitions; do not look them up again.

Add memory updates only for reusable symbols you inspected with repository tools, such as shared config, endpoints, clients, auth, or command wrappers. Format each entry exactly as:

`symbol_name: one-line description of what the symbol does`

## Analysis Rules

- Derive behavior from executable code, server memory, and repository-tool results; do not infer what an in-project symbol does from its name. Write `unknown` when a material detail cannot be resolved.
- List distinct core operations in execution order, ending with what the tool returns. Put each in a `Step` field and the routine work supporting it (setup, input checks, parsing, formatting, retries and reconnects, diagnostic logging, error handling, failure paths) in its `Supporting` field, or `None`. Never fold a critical operation into another step or into `Supporting`; give each critical operation on a distinct resource its own step. The same operation reached from different branches is one step; state its conditions. Operations on the same resource or connection, such as a transaction's begin, query, and commit, form one step.
- In a critical step, `Step` states only the critical operation: the action, the affected resource or boundary, the data and its source (tool input, external response, local state, or constant), any triggering condition, and the validation guarding it, or that none is visible. Do not expand low-level helper calls.

## Output

Return only the following two blocks without a code fence:

<description>
**1. Primary Function**
- Summary: ...

**2. Inputs and Runtime Dependencies**
Tool parameters:
- Name: ...
  Type: ...
  Default: ...
  Constraints: ...
  Role: ...

Runtime dependencies:
- Environment variables: ...
- Config files: ...
- Files or directories: ...
- Network services/endpoints: ...
- Credentials/secrets: ...
- External programs: ...

**3. Operations**
- Step: ...
  Critical: yes/no
  Supporting: ...
</description>
<memory_update>
symbol_name: one-line description of what the symbol does
</memory_update>

`Constraints` lists only restrictions the code itself enforces (checks, normalization, schema validation). Runtime dependencies list names only; describe their use in Operations.

Write `None` for an empty subsection or list, including inside `<memory_update>` when there are no updates.
