Compare the author's description with the code-derived description. Report only meaningful semantic mismatches.

## Inputs

Original description:
{{developer_description}}

Code-derived description:
{{code_derived_description}}

## Task

Perform a bidirectional semantic diff:

1. Every critical code operation must be adequately disclosed by the original description.
2. Every claim in the original description must be supported by the code-derived description.
3. Report contradictions using the applicable `code_extra` or `description_extra` type.

Treat each `Step` under `**3. Operations**` as a code operation. Its `Supporting` field lists routine work for that step: use it as evidence when checking claims, but never report it as a separate `code_extra`. Check parameter claims against each parameter's `Type`, `Default`, and `Constraints`.

## Original-Description Claims

Split the original description into distinct claims of two types:

1. **Functional claims** describe what the tool does and how it behaves, including inputs, outputs, side effects, external interactions, data handling, controls, restrictions, and failures.
2. **Performance claims** describe positive qualities, guarantees, or endorsements that encourage choosing or trusting the tool, such as claims that it is safe, accurate, or official.

Restrictions the description clearly implies, such as read-only access for a tool that queries or reads, are claims too. Guidance addressed to the caller, such as best practices for writing inputs, is not a claim about the tool's behavior.

A performance claim requires concrete code-derived behavior that demonstrates or verifies the claimed property. Names, branding, unsupported assertions, and delegation to an external service are insufficient evidence. Require stronger evidence for absolute terms such as `always`, `never`, `guaranteed`, and `best`.

## Matching Rules

- Treat the original description as a concise summary of the tool's intended behavior, not as an exhaustive implementation trace. Match semantically rather than word for word; a broad functional claim may cover routine implementation steps that directly support the same behavior.
- Do not report routine work, such as the contents of `Supporting` fields, when it is part of a disclosed operation.
- Treat external-service behavior as opaque. If a potential mismatch depends on the service's response or other unverifiable server-side behavior, place it in `unverified_inconsistency` rather than `inconsistency`.
- Security or resource-boundary guarantees require visible enforcement such as validation, normalization, containment, allowlisting, or authorization. A check that crafted input can bypass is not enforcement. Report relevant missing protection when user input reaches a critical sink.

## Materiality and Abstraction

- Report material differences in purpose, observable behavior, effects, security, or explicit guarantees; do not report minor implementation details.
- Treat semantically equivalent representations, wrappers, annotations, and routine branches as consistent unless the original description explicitly promises otherwise.
- Functional claims: report `description_extra` only when the evidence contradicts the claim, or when the claim concerns behavior the code controls and the code demonstrably does not implement it. Missing detail alone is not a contradiction.
- Performance claims and security or restriction guarantees: report `description_extra` when the code-derived description shows no enforcing behavior. A critical step stating that no validation is visible, or a parameter with `Constraints: None`, is such evidence.

## Report Rules

Use these inconsistency types:

- `code_extra`: A critical code behavior or side effect is not adequately disclosed.
- `description_extra`: A functional claim the code contradicts, or a performance or security claim the code does not enforce.

For `code_extra`, report a step marked `Critical: yes` when the original description does not disclose it, either explicitly or as a direct part of a disclosed operation. An operation is not adequately disclosed if unprotected input can make it do more than the description says. Combine items that describe the same underlying mismatch.

Use `unverified_inconsistency` for a potential mismatch whose truth depends on opaque external-service behavior and cannot be established or contradicted from the provided evidence. Use the same item schema and the otherwise applicable `code_extra` or `description_extra` type.

Set `status` from verified `inconsistency` items only:

- `inconsistent` when `inconsistency` is non-empty;
- `clean` when `inconsistency` is empty, even if `unverified_inconsistency` is non-empty.

## Output

Return only valid JSON with this schema:

```json
{
  "status": "clean | inconsistent",
  "inconsistency": [
    {
      "type": "code_extra | description_extra",
      "code": "operation from code-derived description, or NONE",
      "description": "original-description claim, or NONE",
      "reason": "one sentence explaining the mismatch"
    }
  ],
  "unverified_inconsistency": [
    {
      "type": "code_extra | description_extra",
      "code": "operation from code-derived description, or NONE",
      "description": "original-description claim, or NONE",
      "reason": "one sentence explaining the mismatch"
    }
  ]
}
```

If only external-service verification is missing, return for example:

```json
{
  "status": "clean",
  "inconsistency": [],
  "unverified_inconsistency": [
    {
      "type": "description_extra",
      "code": "NONE",
      "description": "Limited to 100 results",
      "reason": "The result limit is controlled by an opaque external service and cannot be verified from the provided evidence."
    }
  ]
}
```

If there are no verified or unverified inconsistencies, return:

```json
{"status": "clean", "inconsistency": [], "unverified_inconsistency": []}
```
