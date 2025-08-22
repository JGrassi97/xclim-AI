# xclim_tools.core.agent
# =================================================

# This module defines the `Xclim_AI` class, an object-oriented wrapper that:
# 1. Initializes valid xclim tools with a given dataset.
# 2. Constructs a LangGraph with two nodes:
#    - a RAG agent that selects relevant indicators
#    - a tool execution agent that runs xclim indicators using LangChain
# 3. Executes the graph with a given input query and returns the results.

# The system supports automatic logging and output export.


import uuid
import re
import types
import contextlib
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from xclim_ai.utils.tools import get_valid_tools
from xclim_ai.datasets import load_dataset_from_config
from xclim_ai.rag.XclimRAGAgent import XclimRAGAgent
from xclim_ai.utils.prompts import load_prompt
from xclim_ai.utils.paths import OUTPUT_RESULTS
from xclim_ai.utils.logging import get_logger, StreamToLogger
from xclim_ai.utils.streaming import get_streamer, EventType


class RagToolState(TypedDict, total=False):
    """Shared LangGraph state between nodes."""
    query: str
    refined_query: str
    indicators: List[Dict[str, Any]]
    tool_input: str
    tool_result: Any


class Xclim_AI:
    """
    Object-oriented wrapper that builds:
      1. the list of valid tools,
      2. the graph nodes ('rag' and 'tool_agent'),
      3. the compiled LangGraph (self.graph).
    """

    def __init__(
        self,
        llm,
        *,
        dataset: str | None = None,
        k: int = 5,
        max_iters: int = 5,
        score_threshold: float = 0.75,
        llm_summary: bool = False,
        verbose: bool = False,
        output_dir: Path | None = None,
        rag_model: str | None = None,
        **ds_kwargs,
    ):
        self.llm = llm
        self.k = k
        self.max_iters = max_iters
        self.score_threshold = score_threshold
        self.llm_summary = llm_summary
        self.verbose = verbose
        self.rag_model = rag_model

        if output_dir:
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_id = uuid.uuid4().hex
            self.output_dir = OUTPUT_RESULTS / run_id
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = get_logger(name=f'XclimAI', log_file=f"{self.output_dir}/agent_debug.log")
        self._streamer = get_streamer()

        
        if dataset is None:
            self.ds, self.ds_params, self.ds_name = load_dataset_from_config(return_params=True, **ds_kwargs)
        else:
            self.ds, self.ds_params, self.ds_name = load_dataset_from_config(name=dataset, return_params=True, **ds_kwargs)

        

        raw_tools = get_valid_tools()
        self.valid_tools = [
            cls(ds=self.ds, llm_summary=self.llm_summary, out_dir=self.output_dir)
            for cls in raw_tools
        ]

        # Wrap tools to emit streaming events with tool name and parameters
        self._wrap_tools_for_streaming()

        self.graph = self._build_graph()

    def run(self, query: str, additional_system: str | None = None) -> RagToolState:
        """Run the LangGraph on a query and return the final state.

        additional_system: optional per-run system instructions to prepend to the tool agent prompt.
        """
        # Store per-run additional system prompt for downstream nodes
        self.additional_system = (additional_system or "").strip()

        self._streamer.emit(EventType.AGENT_START, f"Starting Xclim_AI with query: {query}", "Xclim_AI")
        self._logger.info(f"Query: {query}")

        result = self.graph.invoke({"query": query})

        self._streamer.emit(EventType.AGENT_END, "Xclim_AI execution completed", "Xclim_AI")
        return result

    def _build_graph(self):
        """Construct and compile the LangGraph."""
        
        def _sanitize_params(params: Dict[str, Any]) -> Dict[str, str]:
            SENSITIVE = {"api_key", "token", "password", "secret", "authorization"}
            def _short(v):
                try:
                    if isinstance(v, (list, tuple)):
                        inner = ", ".join(_short(x) for x in list(v)[:4])
                        more = "…" if len(v) > 4 else ""
                        return f"[{inner}{more}]"
                    if isinstance(v, dict):
                        items = list(v.items())[:4]
                        inner = ", ".join(f"{k}={_short(val)}" for k, val in items)
                        more = "…" if len(v) > 4 else ""
                        return f"{{{inner}{more}}}"
                    s = str(v)
                except Exception:
                    s = repr(v)
                return (s[:160] + "…") if len(s) > 160 else s
            out: Dict[str, str] = {}
            for k, v in (params or {}).items():
                key = str(k)
                if key.lower() in SENSITIVE:
                    out[key] = "***"
                else:
                    out[key] = _short(v)
            return out

        def _emit_tool_event(action: str, tool_name: str, params: Dict[str, Any]):
            self._streamer.emit(
                EventType.TOOL_OBSERVATION,
                f"{action.title()} tool: {tool_name}",
                "ToolAgent",
                tool=tool_name,
                params=_sanitize_params(params),
                action=action,
            )

        def rag_node(state: RagToolState) -> RagToolState:
            query = state["query"]
            rag_agent = XclimRAGAgent(
                k=self.k,
                max_iters=self.max_iters,
                score_threshold=self.score_threshold,
                verbose=self.verbose,
                rag_model=self.rag_model,
            )
            refined_query, indicators = rag_agent.run(query)
            return {
                **state,
                "refined_query": refined_query,
                "indicators": indicators,
            }

        def tool_agent_node(state: RagToolState) -> RagToolState:
            top_ids = [x["id"].split(".")[-1] for x in state["indicators"]]
            top_ext = [f"{x['id'].split('.')[-1]}: {x['text']}" for x in state["indicators"]]

            self._streamer.emit(EventType.TOOL_START, 
                               f"Starting tool execution with {len(top_ids)} indicators", 
                               "ToolAgent",
                               indicators=top_ids)
            # Agent is planning which tool to call and with what inputs
            self._streamer.emit(EventType.AGENT_THINKING, "Planning tool calls", "ToolAgent")

            instructions = load_prompt("tool_agent.md")
            instructions = (
                instructions.replace("{variables}", ", ".join(self.ds.data_vars))
                .replace("{top_xclim_ind_to_prompt}", ", ".join(top_ids))
                .replace("{top_xclim_ind_to_prompt_ext}", ", ".join(top_ext))
            )
            # Prepend additional per-run system instructions if provided
            extra_sys = getattr(self, "additional_system", "").strip()
            if extra_sys:
                instructions = f"{extra_sys}\n\n" + instructions

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", instructions),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.valid_tools,
                prompt=prompt,
            )

            executor = AgentExecutor(
                agent=agent,
                tools=self.valid_tools,
                verbose=self.verbose,
                max_iterations=15,
                return_intermediate_steps=True,
            )

            if self.verbose:
                stream_logger = StreamToLogger(self._logger, level=20)  # INFO level
                with contextlib.redirect_stdout(stream_logger):
                    result = executor.invoke({"input": state["query"]})
            else:
                result = executor.invoke({"input": state["query"]})

            self._streamer.emit(EventType.TOOL_END, 
                               "Tool execution completed", 
                               "ToolAgent")
            # Agent is consolidating outputs and deciding the final response
            self._streamer.emit(EventType.AGENT_THINKING, "Synthesizing final answer", "ToolAgent")

            output_md_path = self.output_dir / "final_output.md"
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(result["output"])

            return {**state, "tool_result": result}

        graph = StateGraph(RagToolState)
        graph.add_node("rag", rag_node)
        graph.add_node("tool_agent", tool_agent_node)
        graph.set_entry_point("rag")
        graph.add_edge("rag", "tool_agent")
        graph.set_finish_point("tool_agent")
        return graph.compile()

    def _wrap_tools_for_streaming(self) -> None:
        """Wrap each tool's _run to emit start/end events with name and parameters."""
        def _sanitize_params_local(params: Dict[str, Any]) -> Dict[str, str]:
            SENSITIVE = {"api_key", "token", "password", "secret", "authorization"}
            def _short(v):
                try:
                    if isinstance(v, (list, tuple)):
                        inner = ", ".join(_short(x) for x in list(v)[:4])
                        more = "…" if len(v) > 4 else ""
                        return f"[{inner}{more}]"
                    if isinstance(v, dict):
                        items = list(v.items())[:4]
                        inner = ", ".join(f"{k}={_short(val)}" for k, val in items)
                        more = "…" if len(v) > 4 else ""
                        return f"{{{inner}{more}}}"
                    s = str(v)
                except Exception:
                    s = repr(v)
                return (s[:160] + "…") if len(s) > 160 else s
            out: Dict[str, str] = {}
            for k, v in (params or {}).items():
                key = str(k)
                if key.lower() in SENSITIVE:
                    out[key] = "***"
                else:
                    out[key] = _short(v)
            return out
        for tool in self.valid_tools:
            if not hasattr(tool, "_run"):
                continue
            original_run = tool._run
            # Prefer the explicit tool name; if not reliable, derive from class name
            explicit_name = getattr(tool, 'name', None)
            cls_name = tool.__class__.__name__
            base = re.sub(r"Tool$", "", cls_name)
            # CamelCase -> snake_case
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
            tool_name = explicit_name if (isinstance(explicit_name, str) and explicit_name and explicit_name != "base_indicator_tool") else snake

            def make_wrapper(orig, captured_tool_name: str, captured_class_name: str):
                def _wrapped_run(*args, **kwargs):
                    try:
                        # Emit start event with sanitized params
                        self._streamer.emit(
                            EventType.TOOL_OBSERVATION,
                            f"Start tool: {captured_tool_name}",
                            "ToolAgent",
                            tool=captured_tool_name,
                            tool_class=captured_class_name,
                            params=_sanitize_params_local(kwargs),
                            action="start",
                        )
                    except Exception:
                        pass
                    try:
                        result = orig(*args, **kwargs)
                    finally:
                        try:
                            self._streamer.emit(
                                EventType.TOOL_OBSERVATION,
                                f"End tool: {captured_tool_name}",
                                "ToolAgent",
                                tool=captured_tool_name,
                                tool_class=captured_class_name,
                                params=_sanitize_params_local(kwargs),
                                action="end",
                            )
                        except Exception:
                            pass
                    return result
                return _wrapped_run

            try:
                wrapped = make_wrapper(original_run, tool_name, cls_name)
                tool._run = types.MethodType(wrapped, tool)
            except Exception:
                # If wrapping fails, leave tool as-is
                continue
