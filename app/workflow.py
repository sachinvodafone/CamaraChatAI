from typing import Optional

from app.index import get_index
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.server.api.models import ChatRequest

# New imports for current LlamaIndex version
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate # For the system prompt


def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    index = get_index(chat_request=chat_request)
    if index is None:
        raise RuntimeError(
            "Index not found! Please run `uv run generate` to index the data first."
        )

    # 1. Create the Retriever
    # This retrieves relevant nodes from your index
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5) # You can adjust top_k

    # 2. Configure the Response Synthesizer to handle citations
    # The 'compact' mode often includes sources by default, or you can use
    # other modes like 'tree_summarize' and ensure sources are returned.
    # The actual "citation" formatting logic happens here.
    response_synthesizer = ResponseSynthesizer.from_defaults(
        response_mode="compact", # or "tree_summarize", etc.
        # You might need to explicitly set `text_qa_template` or `refine_template`
        # if you want custom citation prompts within the response synthesizer.
        # For simple source inclusion, `compact` often suffices.
    )

    # 3. Create the Query Engine from the retriever and response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # 4. Wrap the Query Engine in a Tool for the AgentWorkflow
    # ToolMetadata provides the name and description for the agent to use
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_knowledge_base", # Give your tool a descriptive name
            description="Useful for answering questions about the knowledge base.",
        ),
    )

    # Define the system prompt for the agent
    # CITATION_SYSTEM_PROMPT is removed, as citation is now handled by the ResponseSynthesizer.
    # You might want to instruct the LLM to refer to sources in your system prompt.
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
        system_prompt=system_prompt.format(), # Use .format() if it's a PromptTemplate
    )
