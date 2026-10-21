from langgraph.graph import StateGraph, END
from src.application.workflow.state import GraphState
from src.application.workflow.nodes import GraphNodes
from src.rag.vector_store import VectorStoreService
from src.llm.llm_service import LLMService


def build_graph(vector_store: VectorStoreService, llm: LLMService):
    """Linear graph of the clinical pipeline.

    The interactive questionnaire (Glasgow or category questions) is driven by
    the application service, which calls the nodes directly; this graph
    represents the automatic part that runs once the answers are collected.
    """
    nodes = GraphNodes(vector_store, llm)

    workflow = StateGraph(GraphState)

    workflow.add_node("assess_case", nodes.assess_case)
    workflow.add_node("translate_case", nodes.translate_case)
    workflow.add_node("build_rag_query", nodes.build_rag_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("generate_diagnosis", nodes.generate_diagnosis)
    workflow.add_node("evaluate_candidates", nodes.evaluate_candidates)

    workflow.set_entry_point("assess_case")
    workflow.add_edge("assess_case", "translate_case")
    workflow.add_edge("translate_case", "build_rag_query")
    workflow.add_edge("build_rag_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_diagnosis")
    workflow.add_edge("generate_diagnosis", "evaluate_candidates")
    workflow.add_edge("evaluate_candidates", END)

    return workflow.compile()
