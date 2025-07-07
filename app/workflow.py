from typing import Optional

from app.index import get_index
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.server.api.models import ChatRequest

# Updated import for ResponseSynthesizer
from llama_index.core import get_response_synthesizer # <--- CHANGE IS HERE

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate


def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    index = get_index(chat_request=chat_request)
    if index is None:
        raise RuntimeError(
            "Index not found! Please run `uv run generate` to index the data first."
        )

    # 1. Create the Retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5) # You can adjust top_k

    # 2. Configure the Response Synthesizer to handle citations
    # Use the get_response_synthesizer factory function
    response_synthesizer = get_response_synthesizer(
        response_mode="compact", # or "tree_summarize", etc.
        # You can add other configurations here if needed, e.g., llm=Settings.llm
    )

    # 3. Create the Query Engine from the retriever and response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # 4. Wrap the Query Engine in a Tool for the AgentWorkflow
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_knowledge_base",
            description="Useful for answering questions about the knowledge base.",
        ),
    )

    # Define the system prompt for the agent
    system_prompt = PromptTemplate(
        """You are a helpful assistant.
        Always try to answer questions using the provided tools and information.
        When providing an answer, always refer to the source documents if available
        (e.g., "According to Document [X]...").
        """
    )


    return AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_tool],
        llm=Settings.llm,
        system_prompt=system_prompt.format(),
    )
