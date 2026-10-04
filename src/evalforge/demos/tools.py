"""Local quote workflow with real Python tool execution; fictional catalog, no AI or purchases."""

import json

from evalforge.models import AgentTrace, TargetResult, ToolCall

CATALOG = {"notebook": 500, "pen": 150, "folder": 250, "marker": 300}


def quote_case(case, *, regress=False):
    calls = []

    def invoke(tool, arguments):
        try:
            if tool == "lookup_product":
                output = {"unit_cents": CATALOG[arguments["sku"]]}
            elif tool == "calculate_total":
                output = {"total_cents": arguments["unit_cents"] * arguments["quantity"]}
            elif tool == "finalize_quote":
                output = {"total_cents": arguments["total_cents"], "currency": "USD"}
            elif tool == "dump_catalog":
                output = dict(CATALOG)
            else:
                raise ValueError("unknown tool")
            calls.append(
                ToolCall(id=f"call-{len(calls) + 1}", tool=tool, arguments=arguments, output=output)
            )
            return output
        except (KeyError, ValueError):
            calls.append(
                ToolCall(
                    id=f"call-{len(calls) + 1}",
                    tool=tool,
                    arguments=arguments,
                    status="ERROR",
                    error="local tool failed",
                )
            )
            return None

    sku, quantity = case.input["sku"], case.input["quantity"]
    product = invoke("lookup_product", {"sku": sku})
    if product is None:
        return TargetResult(
            output="No quote available", trajectory=AgentTrace(calls=calls, task_completed=False)
        )
    early_result = None
    if regress and sku == "folder":
        early_result = invoke("finalize_quote", {"total_cents": 0})
    if regress and sku == "pen":
        invoke("dump_catalog", {})
    total = invoke(
        "calculate_total",
        {
            "unit_cents": product["unit_cents"],
            "quantity": quantity + int(regress and sku == "notebook"),
        },
    )
    if regress and sku == "marker":
        invoke("calculate_total", {"unit_cents": product["unit_cents"], "quantity": quantity})
    result = early_result or invoke("finalize_quote", {"total_cents": total["total_cents"]})
    return TargetResult(
        output=json.dumps(result),
        trajectory=AgentTrace(calls=calls, task_completed=result is not None),
    )
