from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.graph.nodes import GraphNodes
from src.services.vector_store import VectorStoreService
from src.services.llm_service import LLMService


def build_graph(vector_store: VectorStoreService, llm: LLMService):
    nodes = GraphNodes(vector_store, llm)

    workflow = StateGraph(GraphState)

    workflow.add_node("collect_description", nodes.collect_description)
    workflow.add_node("ask_gcs_question", nodes.ask_gcs_question)
    workflow.add_node("build_rag_query", nodes.build_rag_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("generate_diagnosis", nodes.generate_diagnosis)

    workflow.set_entry_point("collect_description")
    workflow.add_edge("collect_description", "ask_gcs_question")

    def route_after_question(state: GraphState):
        if state.get("phase") == "rag":
            return "build_rag_query"
        return "ask_gcs_question"

    workflow.add_conditional_edges(
        "ask_gcs_question",
        route_after_question,
        {"build_rag_query": "build_rag_query", "ask_gcs_question": "ask_gcs_question"},
    )

    workflow.add_edge("build_rag_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_diagnosis")
    workflow.add_edge("generate_diagnosis", END)

    return workflow.compile()
