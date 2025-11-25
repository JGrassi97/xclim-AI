import os
import io
import base64
import json
import zipfile
import threading
import time
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from xclim_ai.core.agent import Xclim_AI
from xclim_ai.utils.streaming import get_streamer, EventType, StreamEvent

# ---- Helpers (minimal, UI-agnostic) ----
def find_images(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("*.png"))


def read_final_markdown(output_dir: Path) -> str:
    md_path = output_dir / "final_output.md"
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def export_events(events: List[StreamEvent]) -> str:
    payload = [
        {
            "type": e.type.value,
            "message": e.message,
            "timestamp": e.timestamp,
            "metadata": e.metadata,
            "source": e.source,
        }
        for e in events
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _normalize_tool_name(tool: str | None, tool_class: str | None) -> str:
    """Return a readable tool name, preferring the explicit tool name; else derive from class."""
    if tool and tool != "base_indicator_tool":
        return str(tool)
    if not tool_class:
        return str(tool or "unknown")
    base = re.sub(r"Tool$", "", str(tool_class))
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
    return name or str(tool_class)


def format_tool_history(events: List[StreamEvent], limit: int = 30) -> str:
    """Render a compact, impersonal tool history from TOOL_OBSERVATION events."""
    rows = []
    for e in events:
        if e.type == EventType.TOOL_OBSERVATION and isinstance(e.metadata, dict):
            action = str(e.metadata.get("action") or "")
            tool = _normalize_tool_name(
                e.metadata.get("tool"),
                e.metadata.get("tool_class"),
            )
            params = e.metadata.get("params") or {}
            # Only keep a single entry per invocation: show only 'start' events
            if action != "start":
                continue
            if isinstance(params, dict) and params:
                kv = ", ".join(f"{k}={v}" for k, v in list(params.items())[:6])
                params_txt = f"({kv})"
            else:
                params_txt = ""
            rows.append(f"- {tool}{params_txt}")
    if not rows:
        return "- No tool activity yet"
    if len(rows) > limit:
        rows = rows[-limit:]
    return "\n".join(rows)


def build_summary_prompt(
    events: List[StreamEvent],
    query: str,
    lat: float,
    lon: float,
    model: str,
    k: int,
    max_iters: int,
    score_threshold: float,
) -> str:
    """Compose a short, process-only technical summary prompt from events and settings."""
    # Extract process info from events
    topics: List[str] = []
    selected: List[str] = []
    used_inds: List[str] = []
    orig_q: str | None = None
    refined_q: str | None = None
    hits: List[str] = []
    missing: List[str] = []
    tool_steps: List[str] = []
    for e in events:
        meta = e.metadata if isinstance(e.metadata, dict) else {}
        if e.type == EventType.RAG_QUERY_REFINED:
            if meta.get("original"):
                orig_q = str(meta.get("original"))
            if meta.get("refined"):
                refined_q = str(meta.get("refined"))
            topics = list(meta.get("topics", topics) or topics)
        elif e.type == EventType.RAG_RETRIEVAL:
            hits = list(meta.get("hits", hits) or hits)
        elif e.type == EventType.RAG_EVALUATION:
            missing = list(meta.get("remaining_topics", missing) or missing)
        elif e.type == EventType.RAG_AGGREGATION:
            selected = list(meta.get("selected_indicators", selected) or selected)
        elif e.type in (EventType.TOOL_START, EventType.TOOL_END):
            used_inds = list(meta.get("indicators", used_inds) or used_inds)
        elif e.type == EventType.TOOL_OBSERVATION and str(meta.get("action")) == "start":
            tname = str(meta.get("tool") or "unknown")
            params = meta.get("params") or {}
            # Produce a compact "key=value" list
            kv = ", ".join(f"{k}={v}" for k, v in list(params.items())[:6]) if isinstance(params, dict) else str(params)
            step = f"{tname}({kv})" if kv else tname
            tool_steps.append(step)

    # De-duplicate preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            sx = str(x)
            if sx not in seen:
                out.append(sx)
                seen.add(sx)
        return out

    topics = dedup(topics)
    selected = dedup(selected)
    used_inds = dedup(used_inds)
    hits = dedup(hits)
    missing = dedup(missing)

    settings_line = f"Settings: agent_model={model}, top_k={k}, max_iters={max_iters}, score_threshold={score_threshold:.2f}."
    context_lines = [
        f"User question: {query}",
        f"Location: lat={lat:.4f}, lon={lon:.4f}",
        settings_line,
    ]
    if orig_q or refined_q:
        context_lines.append(
            f"Query refinement: original={'“'+orig_q+'”' if orig_q else 'n/a'}; refined={'“'+refined_q+'”' if refined_q else 'n/a'}."
        )
    if topics:
        context_lines.append(f"Topics identified: {', '.join(topics)}.")
    if hits:
        context_lines.append(f"Retrieval candidates: {', '.join(hits[:5])}{'…' if len(hits)>5 else ''}.")
    if missing:
        context_lines.append(f"Coverage check: missing topics = {', '.join(missing)}.")
    if selected:
        context_lines.append(f"Selected indicators: {', '.join(selected)}.")
    if used_inds:
        context_lines.append(f"Executed indicators: {', '.join(used_inds)}.")

    if tool_steps:
        context_lines.append(f"Tool execution steps: {'; '.join(tool_steps[:12])}{'…' if len(tool_steps)>12 else ''}.")
    context = "\n".join(context_lines)
    instructions = (
        "Write a technical, process-only summary in English (150–220 words). "
        "Focus on climate methodology and justify each step of the workflow: configuration, query refinement rationale, retrieval criteria, coverage assessment, indicator selection logic (e.g., relevance/diversification), and execution pipeline (datasets, temporal/spatial aggregation, computed statistics, plotting). "
        "Do not report results, findings, or conclusions; describe only what was done and why."
    )
    return f"{instructions}\n\nProcess context\n-----\n{context}"


class StreamEventCollector:
    """Collects stream events in a thread-safe way."""
    def __init__(self):
        self.events: List[StreamEvent] = []
        self.lock = threading.Lock()

    def add_event(self, event: StreamEvent):
        with self.lock:
            self.events.append(event)

    def get_events(self) -> List[StreamEvent]:
        with self.lock:
            return self.events.copy()

    def clear(self):
        with self.lock:
            self.events.clear()


def current_status(events: List[StreamEvent]) -> Tuple[int, str]:
    """Return (progress 0-100, status label) based on latest high-level events."""
    if not events:
        return 0, "Idle"
    milestones = [
        EventType.RAG_QUERY_REFINED,
        EventType.RAG_RETRIEVAL,
        EventType.RAG_EVALUATION,
        EventType.RAG_AGGREGATION,
        EventType.TOOL_END,
        EventType.AGENT_END,
    ]
    done = sum(1 for m in milestones if any(e.type == m for e in events))
    pct = int(done / len(milestones) * 100)

    priority = [
        EventType.TOOL_START,
        EventType.AGENT_THINKING,
        EventType.RAG_AGGREGATION,
        EventType.RAG_EVALUATION,
        EventType.RAG_RETRIEVAL,
        EventType.RAG_QUERY_REFINED,
        EventType.AGENT_START,
    ]
    label_map = {
        EventType.AGENT_START: "Initializing agent",
        EventType.RAG_QUERY_REFINED: "Refining query",
        EventType.RAG_RETRIEVAL: "Retrieving indicators",
        EventType.RAG_EVALUATION: "Evaluating coverage",
        EventType.RAG_AGGREGATION: "Selecting final indicators",
        EventType.TOOL_START: "Running climate computations",
        EventType.AGENT_THINKING: "Planning next step",
        EventType.TOOL_END: "Computation finished",
        EventType.AGENT_END: "Completed",
    }
    current = None
    for e in reversed(events):
        if e.type in priority or e.type in (EventType.TOOL_END, EventType.AGENT_END):
            current = e
            break
    if not current:
        return pct, "Running…"
    return pct, label_map.get(current.type, current.type.value.replace("_", " ").title())


def status_description(events: List[StreamEvent], label: str) -> str:
    """Return a human-friendly description that adapts to the current label and last event metadata."""
    if not events:
        return "Waiting to start the analysis. Configure settings and press Send."
    priority = [
        EventType.TOOL_START,
        EventType.RAG_AGGREGATION,
        EventType.RAG_EVALUATION,
        EventType.RAG_RETRIEVAL,
        EventType.RAG_QUERY_REFINED,
        EventType.AGENT_START,
        EventType.TOOL_END,
        EventType.AGENT_END,
    ]
    last = None
    for e in reversed(events):
        if e.type in priority:
            last = e
            break
    if last is None:
        last = events[-1]

    def list_to_text(items: List[str] | None, max_items: int = 5) -> str:
        if not items:
            return "none"
        items = [str(i) for i in items]
        if len(items) <= max_items:
            return ", ".join(items)
        return ", ".join(items[:max_items]) + f" and {len(items)-max_items} more"

    if "Refining query" in label:
        orig = last.metadata.get("original") if isinstance(last.metadata, dict) else None
        refined = last.metadata.get("refined") if isinstance(last.metadata, dict) else None
        topics = last.metadata.get("topics") if isinstance(last.metadata, dict) else None
        return (
        "Refining the question to better match climate indicators. "
            + (f"Original: “{orig}”. " if orig else "")
            + (f"Refined: “{refined}”. " if refined else "")
            + (f"Identified topics: {list_to_text(topics)}." if topics else "")
        ).strip()
    if "Retrieving indicators" in label:
        hits = last.metadata.get("hits") if isinstance(last.metadata, dict) else None
        return (
        "Searching the knowledge base for the most relevant climate indicators "
            + (f"(top matches: {list_to_text(hits)})" if hits else "")
        ).strip()
    if "Evaluating coverage" in label:
        rem = last.metadata.get("remaining_topics") if isinstance(last.metadata, dict) else None
        return (
        "Checking whether retrieved indicators fully cover the request "
            + (f"(missing: {list_to_text(rem)})" if rem else "")
        ).strip()
    if "Selecting final indicators" in label:
        sel = last.metadata.get("selected_indicators") if isinstance(last.metadata, dict) else None
        return (
        "Selecting a compact set of indicators that best represent the query "
            + (f"(selected: {list_to_text(sel)})" if sel else "")
        ).strip()
    if "Planning next step" in label:
        # Differentiate thinking phases: before first tool vs post-tools synthesis
        # Look up the latest AGENT_THINKING event to tailor the message
        thinking_detail = None
        for e in reversed(events):
            if e.type == EventType.AGENT_THINKING:
                thinking_detail = e.message
                break
        if thinking_detail and "Synthesizing" in thinking_detail:
            return "Consolidating intermediate outputs and preparing the final response."
        return "Assessing whether to call another tool or proceed to answer."
    if "Running climate computations" in label:
        inds = last.metadata.get("indicators") if isinstance(last.metadata, dict) else None
        # Try to show the current tool and parameters from the latest TOOL_OBSERVATION(start)
        current_tool = None
        for e in reversed(events):
            if e.type == EventType.TOOL_OBSERVATION and isinstance(e.metadata, dict) and str(e.metadata.get("action")) == "start":
                tname = _normalize_tool_name(
                    e.metadata.get("tool"),
                    e.metadata.get("tool_class"),
                )
                params = e.metadata.get("params") or {}
                if isinstance(params, dict) and params:
                    kv = ", ".join(f"{k}={v}" for k, v in list(params.items())[:6])
                    current_tool = f"{tname}({kv})"
                else:
                    current_tool = tname
                break
        base = "Executing selected indicators on climate data: computing time series, statistics, and plots"
        if inds:
            base += f" (indicators: {list_to_text(inds)})"
        if current_tool:
            base += f" — current tool: {current_tool}"
        return base.strip()
    if "Computation finished" in label or "Completed" in label:
        return "Analysis completed. The final report and figures are ready below."
    return "Running the analysis pipeline…"

# Minimal CSS only for spinner animation; rely on Streamlit's dark theme for the rest
st.markdown(
    """
    <style>
    /* Nicer, accessible spinner: three-dot pulse that adapts to current color */
    .spinner{display:inline-flex;align-items:center;gap:6px}
    .spinner .dot{width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.35;animation:blink 1.2s infinite ease-in-out}
    .spinner .dot:nth-child(2){animation-delay:.2s}
    .spinner .dot:nth-child(3){animation-delay:.4s}
    @keyframes blink{0%,80%,100%{opacity:.35}40%{opacity:1}}
    .status{display:flex;align-items:center;gap:10px}
    .status-desc{opacity:.9}
    </style>
    """,
    unsafe_allow_html=True,
)
st.set_page_config(page_title="xclim-AI", layout="wide")
st.title("xclim-AI – Climate Indicator Agent")

with st.sidebar:
    st.header("Agent Settings")
    # Advanced settings in expander
    with st.expander("Model and retrieval", expanded=False):
        provider = st.selectbox(
            "Provider",
            options=["openai", "azure-openai", "ollama"],
            index=0,
            help="Must match your credentials in config.yaml (for ollama a local daemon must be running)",
        )

        @st.cache_data(show_spinner=False)
        def _list_ollama_models() -> list[str]:
            try:
                proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "ollama list failed")
                lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
                models: list[str] = []
                for ln in lines:
                    if ln.lower().startswith("name ") or ln.startswith("NAME "):
                        continue
                    # Format usually: name  size   modified
                    name = ln.split()[0]
                    models.append(name)
                return models or ["llama3"]
            except Exception:
                return ["llama3"]

        if provider == "ollama":
            available_models = _list_ollama_models()
            model = st.selectbox(
                "Ollama model",
                options=available_models,
                index=0,
                help="Models detected via 'ollama list'",
            )
        else:
            model = st.selectbox(
                "GPT model",
                options=[
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-4.1-mini",
                    "gpt-4.1",
                    "o4-mini",
                    "o4",
                ],
                index=0,
                help="Overrides model from config for this session",
            )
        k = st.slider("Top-K indicators", min_value=1, max_value=10, value=3, step=1)
        max_iters = st.slider("Max RAG iterations", min_value=1, max_value=10, value=3, step=1)
        score_threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        llm_summary = st.checkbox("LLM summary", value=True)
        verbose = st.checkbox("Verbose (console logs)", value=True)

    st.markdown("---")


st.subheader("💬 Analysis")
form_container = st.container()
with form_container:
    with st.form("analysis_input"):
        col_q, col_lat, col_lon = st.columns([3, 1, 1])
        with col_q:
            query = st.text_area(
                "Your question",
                value="Heat waves and drought conditions in the next decades",
                height=120,
                help="Describe the climate question you want to analyze.",
            )
        with col_lat:
            lat = st.number_input("Lat", value=45.0, format="%.6f")
        with col_lon:
            lon = st.number_input("Lon", value=10.0, format="%.6f")
        additional_info = st.text_area(
            "Additional information (system instructions, optional)",
            value="",
            height=100,
            help="Any constraints or formatting guidelines for the model (e.g., output style, assumptions to keep in mind).",
        )
        submitted = st.form_submit_button("Run analysis")

if submitted:
    with st.spinner("Running agent… this may take a few minutes on first run"):
        # Init session-state for event collection
        if "event_collector" not in st.session_state:
            st.session_state.event_collector = StreamEventCollector()
        # Reset for this run
        st.session_state.event_collector.clear()
        # Initialize throttling state for the status description
        st.session_state._desc_last_label = None
        st.session_state._desc_last_text = None
        st.session_state._desc_last_ts = 0.0

        try:
            if provider == "ollama":
                try:
                    from langchain_community.chat_models import ChatOllama  # type: ignore
                except Exception as _imp_err:  # pragma: no cover
                    st.error(f"Missing Ollama integration: {_imp_err}")
                    st.stop()
                llm = ChatOllama(model=model, temperature=0)
            else:
                from xclim_ai.utils.llm import initialize_llm  # noqa: WPS433
                llm = initialize_llm(model=model)
        except Exception as e:
            st.error(f"Failed to initialize LLM. Error: {e}")
            st.stop()

        try:
            agent = Xclim_AI(
                llm=llm,
                lat=lat,
                lon=lon,
                k=k,
                max_iters=max_iters,
                score_threshold=score_threshold,
                llm_summary=llm_summary,
                verbose=verbose,
            )
        except Exception as e:
            st.error(f"Failed to initialize agent or dataset. Error: {e}")
            st.stop()

        # Attach event listener to the global streamer
        streamer = get_streamer()
        streamer.add_listener(st.session_state.event_collector.add_event)

        # Run the agent in a background thread while streaming events live
        result_box = {"value": None, "error": None}

        def _run_agent():
            try:
                result_box["value"] = agent.run(query, additional_system=additional_info)
            except Exception as exc:
                result_box["error"] = exc

        th = threading.Thread(target=_run_agent, daemon=True)
        th.start()

    # Streaming response containers
    status_placeholder = st.empty()
    desc_placeholder = st.empty()
    # Live tool history panel
    with st.expander("Tool history", expanded=False):
        tool_hist_placeholder = st.empty()

    # While the agent runs, poll events and update the UI (bottom-up)
    while th.is_alive():
        events = st.session_state.event_collector.get_events()
        if events:
            pct, label = current_status(events)
            # Status bar with spinner and dynamic description
            status_html = f"<div class='status'><span class='spinner'><span class='dot'></span><span class='dot'></span><span class='dot'></span></span><strong>{label}</strong><span style='margin-left:auto'>{pct}%</span></div>"
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            # Throttle description changes to at most once every 2 seconds
            now = time.time()
            current_text = status_description(events, label)
            if st.session_state._desc_last_label is None:
                st.session_state._desc_last_label = label
                st.session_state._desc_last_text = current_text
                st.session_state._desc_last_ts = now
                desc_placeholder.markdown(f"<div class='status-desc'>{current_text}</div>", unsafe_allow_html=True)
            else:
                changed = (label != st.session_state._desc_last_label) or (current_text != st.session_state._desc_last_text)
                if changed and (now - st.session_state._desc_last_ts) < 2.0:
                    # Keep previous description until 2s have elapsed
                    desc_placeholder.markdown(f"<div class='status-desc'>{st.session_state._desc_last_text}</div>", unsafe_allow_html=True)
                elif changed:
                    # Apply the new description and reset timer
                    st.session_state._desc_last_label = label
                    st.session_state._desc_last_text = current_text
                    st.session_state._desc_last_ts = now
                    desc_placeholder.markdown(f"<div class='status-desc'>{current_text}</div>", unsafe_allow_html=True)
                else:
                    # No change
                    desc_placeholder.markdown(f"<div class='status-desc'>{st.session_state._desc_last_text}</div>", unsafe_allow_html=True)
                # Update tool history
                tool_hist_placeholder.markdown(format_tool_history(events))
        time.sleep(0.4)

    # Final update after thread completion
    events = st.session_state.event_collector.get_events()
    if events:
        pct, label = current_status(events)
        # Completed state (no spinner)
        status_html = f"<div class='status'><strong>{'✅ ' if pct==100 else ''}{label}</strong><span style='margin-left:auto'>{pct}%</span></div>"
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        # Final description update (force update, past throttling window)
        final_text = status_description(events, label)
        st.session_state._desc_last_label = label
        st.session_state._desc_last_text = final_text
        st.session_state._desc_last_ts = time.time()
        desc_placeholder.markdown(f"<div class='status-desc'>{final_text}</div>", unsafe_allow_html=True)
    # Final tool history update
    tool_hist_placeholder.markdown(format_tool_history(events))

    # Detach event listener
    streamer.remove_listener(st.session_state.event_collector.add_event)

    # Handle agent result or error
    if result_box["error"] is not None:
        st.error(f"Agent run failed. Error: {result_box['error']}")
        events = st.session_state.event_collector.get_events()
        if events:
            with st.expander("Raw events (debug)"):
                payload = [
                    {
                        "type": e.type.value,
                        "message": e.message,
                        "timestamp": e.timestamp,
                        "metadata": e.metadata,
                        "source": e.source,
                    }
                    for e in events
                ]
                st.json(payload)
        st.stop()
    result = result_box["value"]

    # Final message + images and a single export button
    st.markdown("---")
    final_md = read_final_markdown(agent.output_dir)
    if final_md:
        st.write(final_md)

    imgs = find_images(agent.output_dir)
    if imgs:
        st.markdown("### Figures")
        grid_cols = st.columns(2)
        for i, img in enumerate(imgs):
            with grid_cols[i % 2]:
                st.image(str(img), caption=img.name, use_container_width=True)

    # Single export button: report+images+events
    events = st.session_state.event_collector.get_events()
    ev_json = export_events(events).encode("utf-8")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if final_md:
            zf.writestr("final_report.md", final_md)
        for p in imgs:
            zf.write(p, arcname=f"figures/{p.name}")
        zf.writestr("events.json", ev_json)
    zip_buf.seek(0)
    st.download_button(
        label="Download all results (.zip)",
        data=zip_buf.read(),
        file_name=f"xclim-ai_results_{agent.output_dir.name}.zip",
        mime="application/zip",
    )

    # AI summary expander using a small LLM (gpt-4.1-mini), cached per run
    st.markdown("### AI summary")
    with st.expander("AI summary (gpt-4.1)"):
        run_id = agent.output_dir.name
        if "ai_summaries" not in st.session_state:
            st.session_state.ai_summaries = {}
        cached = st.session_state.ai_summaries.get(run_id)
        if cached:
            st.write(cached)
        else:
            try:
                from xclim_ai.utils.llm import initialize_llm as _init_llm  # lazy import
                # Prefer a larger model for better technical writing; if unavailable (e.g., Azure), fall back to main llm
                try:
                    small_llm = _init_llm(model="gpt-4.1")
                except Exception:
                    small_llm = llm

                summary_prompt = build_summary_prompt(
                    events=events,
                    query=query,
                    lat=lat,
                    lon=lon,
                    model=model,
                    k=k,
                    max_iters=max_iters,
                    score_threshold=score_threshold,
                )
                # Pass a single concise prompt (no need for explicit chat roles here)
                resp = small_llm.invoke("You are a climate analysis assistant.\n\n" + summary_prompt)
                summary_text = getattr(resp, "content", str(resp))
                st.session_state.ai_summaries[run_id] = summary_text
                st.write(summary_text)
            except Exception as e:
                st.warning(f"AI summary unavailable: {e}")

    # DOCX export (final output + figures + AI summary)
    try:
        from xclim_ai.utils.docx_export import build_docx_report
        docx_meta = {
            "Question": query,
            "Location": f"lat={lat:.4f}, lon={lon:.4f}",
            "Settings": f"agent_model={model}, top_k={k}, max_iters={max_iters}, score_threshold={score_threshold:.2f}",
        }
        ai_summary_text = None
        if "ai_summaries" in st.session_state:
            ai_summary_text = st.session_state.ai_summaries.get(agent.output_dir.name)
        docx_bytes = build_docx_report(
            title="xclim-AI – Climate Indicator Report",
            meta=docx_meta,
            final_markdown=final_md or "",
            images=imgs,
            ai_summary=ai_summary_text or "",
        )
        st.download_button(
            label="Download Word report (.docx)",
            data=docx_bytes,
            file_name=f"xclim-ai_report_{agent.output_dir.name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as e:
        st.info("Optional DOCX export not available. Install 'python-docx' to enable this feature.")

else:
    st.info("Set agent settings from the sidebar, then write your question and press Send.")
