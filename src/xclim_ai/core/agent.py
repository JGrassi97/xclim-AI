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
        output_dir: Path = None,
        **ds_kwargs,
    ):
        self.llm = llm
        self.k = k
        self.max_iters = max_iters
        self.score_threshold = score_threshold
        self.llm_summary = llm_summary
        self.verbose = verbose

        if output_dir:
            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_id = uuid.uuid4().hex
            self.output_dir = OUTPUT_RESULTS / run_id
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = get_logger(name=f'XclimAI', log_file=f"{self.output_dir}/agent_debug.log")

        
        if dataset is None:
            self.ds, self.ds_params, self.ds_name = load_dataset_from_config(return_params=True, **ds_kwargs)
        else:
            self.ds, self.ds_params, self.ds_name = load_dataset_from_config(name=dataset, return_params=True, **ds_kwargs)

        

        raw_tools = get_valid_tools()
        self.valid_tools = [
            cls(ds=self.ds, llm_summary=self.llm_summary, out_dir=self.output_dir)
            for cls in raw_tools
        ]

        self.graph = self._build_graph()

    def run(self, query: str) -> RagToolState:
        """Run the LangGraph on a query and return the final state."""

        self._logger.info(f"Query: {query}")
        return self.graph.invoke({"query": query})

    def _build_graph(self):
        """Construct and compile the LangGraph."""

        def rag_node(state: RagToolState) -> RagToolState:
            query = state["query"]
            rag_agent = XclimRAGAgent(
                k=self.k,
                max_iters=self.max_iters,
                score_threshold=self.score_threshold,
                verbose=self.verbose,
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

            instructions = load_prompt("tool_agent.md")
            instructions = (
                instructions.replace("{variables}", ", ".join(self.ds.data_vars))
                .replace("{top_xclim_ind_to_prompt}", ", ".join(top_ids))
                .replace("{top_xclim_ind_to_prompt_ext}", ", ".join(top_ext))
            )

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
