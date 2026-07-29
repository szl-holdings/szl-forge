"""Professional read-only showcase for the SZL Forge evidence kernel."""
from __future__ import annotations

import html
import json
import os

import gradio as gr

from forge_lab import (
    curriculum_rows,
    decision_counts,
    evaluation_rows,
    formula_rows,
    get_curriculum,
    get_curriculum_stage,
    get_evaluation,
    get_formula,
    get_formulas,
    get_integrity,
    get_receipt,
    get_source,
    get_sources,
    get_status,
    integrity_rows,
    inventory,
    metrics,
    source_rows,
)
from forge_runtime_contract import register_contract_routes


CSS = """
:root {
  --forge-ink: #e8f4f3;
  --forge-muted: #9fb4b1;
  --forge-panel: rgba(13, 25, 27, .78);
  --forge-line: rgba(104, 232, 199, .20);
  --forge-green: #68e8c7;
  --forge-amber: #f2c572;
}
.gradio-container {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 1380px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(circle at 7% 3%, rgba(32, 178, 150, .16), transparent 31rem),
    radial-gradient(circle at 95% 0%, rgba(36, 91, 151, .13), transparent 28rem),
    #071011 !important;
  color: var(--forge-ink) !important;
}
.gradio-container > .main,
.gradio-container .wrap,
.gradio-container main.contain {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
}
.forge-hero {
  border: 1px solid var(--forge-line);
  border-radius: 22px;
  padding: 34px 36px;
  margin: 12px 0 18px;
  background: linear-gradient(135deg, rgba(12, 32, 31, .96), rgba(7, 16, 17, .92));
  box-shadow: 0 24px 70px rgba(0,0,0,.26);
}
.forge-eyebrow { color: var(--forge-green); font: 650 12px/1.2 ui-monospace, monospace; letter-spacing: .14em; text-transform: uppercase; }
.forge-hero h1 { margin: 12px 0 10px; color: #f3fffc; font-size: clamp(2.2rem, 5vw, 4.4rem); letter-spacing: -.055em; line-height: .98; }
.forge-hero p { max-width: 820px; margin: 0; color: #b8cdca; font-size: 1.06rem; line-height: 1.65; }
.forge-state { display: inline-flex; align-items: center; gap: 8px; margin-top: 22px; padding: 7px 12px; border: 1px solid rgba(104,232,199,.3); border-radius: 99px; color: #baf8e8; background: rgba(104,232,199,.08); font: 600 12px/1 ui-monospace, monospace; }
.forge-state:before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--forge-green); box-shadow: 0 0 14px var(--forge-green); }
.forge-metrics { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 10px; margin: 0 0 20px; }
.forge-metric { padding: 16px; border: 1px solid var(--forge-line); border-radius: 14px; background: var(--forge-panel); min-height: 92px; }
.forge-metric span { display: block; color: var(--forge-muted); font: 600 10px/1.2 ui-monospace, monospace; letter-spacing: .09em; text-transform: uppercase; }
.forge-metric strong { display: block; margin-top: 11px; color: #effffb; font: 650 18px/1.25 ui-monospace, monospace; overflow-wrap: anywhere; }
.forge-callout { border-left: 3px solid var(--forge-amber); padding: 10px 14px; margin: 4px 0 16px; background: rgba(242,197,114,.07); color: #dbcda9; }
.forge-section-note { color: var(--forge-muted); font-size: .94rem; line-height: 1.55; }
.forge-api code { color: #baf8e8 !important; }
footer { display: none !important; }
@media (max-width: 980px) { .forge-metrics { grid-template-columns: repeat(3,1fr); } }
@media (max-width: 620px) { .forge-hero { padding: 25px 20px; } .forge-metrics { grid-template-columns: repeat(2,1fr); } }
"""


def _hero() -> str:
    return """
    <section class="forge-hero">
      <div class="forge-eyebrow">SZL Holdings · evidence kernel</div>
      <h1>Forge Lab</h1>
      <p>A provenance-first inspection surface for model configuration, deterministic evaluation receipts, formula metadata, and governed scientific curriculum.</p>
      <div class="forge-state">READ-ONLY · SNAPSHOT EVIDENCE</div>
    </section>
    """


def _metric_cards() -> str:
    values = metrics()
    cards = [
        ("Training state", values["training"]),
        ("Formula records", values["formulas"]),
        ("Governed sources", values["sources"]),
        ("Curriculum stages", values["stages"]),
        ("Fixture checks", values["eval_cases"]),
        ("Artifact integrity", values["integrity"]),
    ]
    rendered = "".join(
        f'<div class="forge-metric"><span>{html.escape(str(label))}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in cards
    )
    return f'<section class="forge-metrics">{rendered}</section>'


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


INVENTORY = inventory()
FORMULA_STATUS_CHOICES = ["ALL", *INVENTORY["formula_statuses"]]
SOURCE_DOMAIN_CHOICES = ["ALL", *INVENTORY["source_domains"]]
SOURCE_DECISION_CHOICES = ["ALL", *INVENTORY["source_decisions"]]


with gr.Blocks(title="SZL Forge Lab") as demo:
    gr.HTML(_hero())
    gr.HTML(_metric_cards())
    gr.HTML(
        '<div class="forge-callout"><strong>Evidence boundary.</strong> '
        "This Space does not claim a completed training run, measured model weights, "
        "frontier benchmark performance, scientific discovery, or mathematical proof.</div>"
    )

    with gr.Tabs():
        with gr.Tab("System status"):
            gr.Markdown(
                "### One contract, separate states\n"
                "Transport reachability, evidence quality, training state, evaluation scope, "
                "and promotion authority are reported independently."
            )
            refresh_status = gr.Button("Refresh packaged evidence", variant="primary")
            with gr.Row():
                status_json = gr.JSON(value=get_status(), label="Status contract")
                integrity_json = gr.JSON(value=get_integrity(), label="Integrity receipt")
            integrity_table = gr.Dataframe(
                value=integrity_rows(),
                headers=["Declared input", "Packaged asset", "State", "Expected SHA-256", "Actual SHA-256"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Manifest-to-Space byte checks",
            )
            gr.Markdown(
                "<span class='forge-section-note'>A matching hash proves byte identity only. "
                "Training data is intentionally not exposed through this showcase.</span>"
            )

        with gr.Tab("Evaluation & receipt"):
            gr.Markdown(
                "### Deterministic contract checks\n"
                "The packaged receipt scores four recorded example responses. A 4/4 result is "
                "evidence about those fixtures—not a model benchmark or generalization claim."
            )
            with gr.Row():
                refresh_eval = gr.Button("Inspect evaluation", variant="primary")
                verify_receipt_button = gr.Button("Verify receipt hash")
            evaluation_table = gr.Dataframe(
                value=evaluation_rows(),
                headers=["Case", "Category", "Result", "Response SHA-256"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Recorded fixture results",
            )
            with gr.Row():
                evaluation_json = gr.JSON(value=get_evaluation(), label="Scoped evaluation")
                receipt_json = gr.JSON(value=get_receipt(), label="Hash-verifiable receipt")

        with gr.Tab("Formula registry"):
            gr.Markdown(
                "### Search the formula index\n"
                "Statuses shown here are declarations from the packaged registry. This Space "
                "does not execute Lean or independently elevate a formula to proven."
            )
            with gr.Row():
                formula_query = gr.Textbox(label="Search", placeholder="ID, title, theorem, or runtime gate")
                formula_status = gr.Dropdown(FORMULA_STATUS_CHOICES, value="ALL", label="Declared status")
                formula_limit = gr.Slider(1, 100, value=25, step=1, label="Maximum results")
            search_formulas = gr.Button("Search registry", variant="primary")
            formula_table = gr.Dataframe(
                value=formula_rows(),
                headers=["ID", "Title", "Declared status", "Lean theorem"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Formula metadata",
            )
            formula_results = gr.JSON(value=get_formulas(), label="Formula query receipt")
            with gr.Row():
                formula_id = gr.Dropdown(INVENTORY["formula_ids"], label="Inspect formula ID")
                inspect_formula = gr.Button("Inspect formula")
            formula_detail = gr.JSON(value={}, label="Formula record")

        with gr.Tab("Scientific source ledger"):
            gr.Markdown(
                "### Default-deny source governance\n"
                "Readable or public does not automatically mean training-compatible. Each source "
                "records a decision, license expression, pin strategy, attribution, privacy boundary, and risk."
            )
            gr.JSON(value=decision_counts(), label="Decision distribution")
            with gr.Row():
                source_domain = gr.Dropdown(SOURCE_DOMAIN_CHOICES, value="ALL", label="Domain")
                source_decision = gr.Dropdown(SOURCE_DECISION_CHOICES, value="ALL", label="Policy decision")
                source_limit = gr.Slider(1, 100, value=50, step=1, label="Maximum results")
            search_sources = gr.Button("Filter source ledger", variant="primary")
            source_table = gr.Dataframe(
                value=source_rows(),
                headers=["Source ID", "Name", "Domain", "Decision", "License", "Risk"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Governed source records",
            )
            source_results = gr.JSON(value=get_sources(), label="Source query receipt")
            with gr.Row():
                source_id = gr.Dropdown(INVENTORY["source_ids"], label="Inspect source ID")
                inspect_source = gr.Button("Inspect source")
            source_detail = gr.JSON(value={}, label="Source record and use boundary")

        with gr.Tab("Curriculum blueprint"):
            gr.Markdown(
                "### Seven governed stages\n"
                "The curriculum connects epistemics, formal math, physics, quantum computation, "
                "neuroscience, SZL formulas, and bounded research agents. It remains a blueprint—not a training claim."
            )
            refresh_curriculum = gr.Button("Inspect curriculum", variant="primary")
            curriculum_table = gr.Dataframe(
                value=curriculum_rows(),
                headers=["Stage", "Domains", "Goal", "Tasks", "Evaluations"],
                datatype=["str", "str", "str", "number", "number"],
                interactive=False,
                wrap=True,
                label="Curriculum stages",
            )
            curriculum_json = gr.JSON(value=get_curriculum(), label="Curriculum contract")
            with gr.Row():
                stage_id = gr.Dropdown(INVENTORY["stage_ids"], label="Inspect stage")
                inspect_stage = gr.Button("Inspect stage")
            stage_detail = gr.JSON(value={}, label="Stage tasks, evaluations, and exit gate")

        with gr.Tab("Callable API"):
            gr.Markdown(
                "### Read-only named endpoints\n"
                "Every endpoint returns packaged evidence only. Calls cannot train, publish, deploy, "
                "promote, download scientific data, or mutate an external system."
            )
            gr.Dataframe(
                value=[
                    ["/status", "System state and explicit limits", "None"],
                    ["/integrity", "Manifest-to-Space SHA-256 checks", "None"],
                    ["/evaluation", "Scoped fixture evaluation", "None"],
                    ["/receipt", "Receipt plus canonical-hash verification", "None"],
                    ["/formulas", "Filtered formula metadata", "query, status, limit"],
                    ["/formula", "One formula record", "formula_id"],
                    ["/sources", "Filtered source-policy metadata", "domain, decision, limit"],
                    ["/source", "One source policy record", "source_id"],
                    ["/curriculum", "Curriculum summary and hard gates", "None"],
                    ["/curriculum-stage", "One full curriculum stage", "stage_id"],
                ],
                headers=["api_name", "Returns", "Inputs"],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Public contract",
            )
            gr.Code(
                value=(
                    "from gradio_client import Client\n\n"
                    "client = Client(\"SZLHOLDINGS/szl-forge-lab\")\n"
                    "status = client.predict(api_name=\"/status\")\n"
                    "formulas = client.predict(\"lambda\", \"ALL\", 10, api_name=\"/formulas\")\n"
                    "sources = client.predict(\"quantum\", \"ALL\", 10, api_name=\"/sources\")"
                ),
                language="python",
                label="Python client example",
            )
            gr.Code(value=_json(get_status()), language="json", label="Example /status response")

    gr.Markdown(
        "---\n**Purpose** inspect · **Try** query · **Evidence** hash and receipt · "
        "**Limits** read-only snapshot · **Reproduce** run Forge manifest, eval, and replay locally"
    )

    # Each named event is a public, read-only Gradio API endpoint. Secondary
    # events update the human-readable tables without adding duplicate endpoints.
    refresh_status.click(
        get_status,
        outputs=status_json,
        api_name="status",
        api_description="Return the read-only Forge Lab system and evidence states.",
    )
    refresh_status.click(
        get_integrity,
        outputs=integrity_json,
        api_name="integrity",
        api_description="Verify packaged assets against run-manifest SHA-256 declarations.",
    )
    refresh_status.click(
        integrity_rows,
        outputs=integrity_table,
        api_name="refresh-integrity-table",
        api_visibility="private",
    )

    refresh_eval.click(
        get_evaluation,
        outputs=evaluation_json,
        api_name="evaluation",
        api_description="Return the explicitly scoped deterministic fixture evaluation.",
    )
    refresh_eval.click(
        evaluation_rows,
        outputs=evaluation_table,
        api_name="refresh-evaluation-table",
        api_visibility="private",
    )
    verify_receipt_button.click(
        get_receipt,
        outputs=receipt_json,
        api_name="receipt",
        api_description="Return the evaluation receipt and canonical SHA-256 verification.",
    )

    search_formulas.click(
        get_formulas,
        inputs=[formula_query, formula_status, formula_limit],
        outputs=formula_results,
        api_name="formulas",
        api_description="Filter read-only formula registry metadata.",
    )
    search_formulas.click(
        formula_rows,
        inputs=[formula_query, formula_status, formula_limit],
        outputs=formula_table,
        api_name="refresh-formula-table",
        api_visibility="private",
    )
    inspect_formula.click(
        get_formula,
        inputs=formula_id,
        outputs=formula_detail,
        api_name="formula",
        api_description="Return one formula registry record without re-proving it.",
    )

    search_sources.click(
        get_sources,
        inputs=[source_domain, source_decision, source_limit],
        outputs=source_results,
        api_name="sources",
        api_description="Filter the packaged scientific source-policy ledger.",
    )
    search_sources.click(
        source_rows,
        inputs=[source_domain, source_decision, source_limit],
        outputs=source_table,
        api_name="refresh-source-table",
        api_visibility="private",
    )
    inspect_source.click(
        get_source,
        inputs=source_id,
        outputs=source_detail,
        api_name="source",
        api_description="Return one source policy, attribution, privacy, and license record.",
    )

    refresh_curriculum.click(
        get_curriculum,
        outputs=curriculum_json,
        api_name="curriculum",
        api_description="Return the governed curriculum blueprint and hard gates.",
    )
    refresh_curriculum.click(
        curriculum_rows,
        outputs=curriculum_table,
        api_name="refresh-curriculum-table",
        api_visibility="private",
    )
    inspect_stage.click(
        get_curriculum_stage,
        inputs=stage_id,
        outputs=stage_detail,
        api_name="curriculum-stage",
        api_description="Return one curriculum stage with tasks, evaluations, and exit gate.",
    )


if __name__ == "__main__":
    try:
        server_app, _, _ = demo.launch(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", "7860")),
            show_error=False,
            prevent_thread_lock=True,
            # The managed SSR shell only packages Brotli variants for its /_app CSS.
            # Use Gradio's client shell so assets remain reachable without relying on
            # proxy content-encoding negotiation.
            ssr_mode=False,
            css=CSS,
            theme=gr.themes.Base(),
        )
        register_contract_routes(server_app, get_status)
    except Exception:
        demo.close()
        raise
    demo.block_thread()
