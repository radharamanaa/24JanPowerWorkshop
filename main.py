"""
Agentic RAG for HR Policy Q&A - Main Application

This module implements an intelligent question-answering system using Azure OpenAI agents
and Retrieval-Augmented Generation (RAG) to answer questions about HR policies.

Architecture:
    1. Document Processing: PDF documents are converted to chunks and embedded
    2. Vector Storage: Embeddings are stored in Qdrant for semantic search
    3. Agent System: Azure OpenAI agent with tool-calling capabilities
    4. RAG Pipeline: Agent retrieves relevant context and generates grounded answers

Key Components:
    - AzureOpenAIChatClient: Azure OpenAI client for agent creation
    - Qdrant: Vector database for semantic search
    - Agent Tools: Custom search tool for retrieving relevant document chunks
    - Structured Output: Pydantic models for response validation

Environment Variables Required:
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    - AZURE_AI_MODEL_DEPLOYMENT_NAME: Your Azure OpenAI deployment name

Usage:
    1. Ensure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)
    2. Set environment variables in .env file
    3. Run prepare_data() once to ingest HR policy documents
    4. Run main() to query the agent

Example:
    $ python main.py
"""

from typing import Final

from agent_framework import ai_function
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from pydantic import Field, BaseModel

from tools import get_prepared_collection
from utils import convert_pdf_to_chunks, create_qdrant_collection, get_embeddings_and_text, ingest_vectors_in_qdrant

# Load environment variables from .env file
load_dotenv()

import os

# Load Azure OpenAI configuration from environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
chat_deployment_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

# Initialize Azure OpenAI client with CLI credentials
# This uses Azure CLI authentication for secure access to Azure OpenAI services
client = AzureOpenAIChatClient(
    credential=AzureCliCredential(),
    endpoint=endpoint,
    deployment_name=chat_deployment_name
)

# Qdrant collection name for storing HR policy embeddings
COLLECTION_NAME: Final[str] = "agentic_ai_power_workshop"

def prepare_data():
    """
    Prepare and ingest HR policy data into Qdrant vector database.

    This function performs the complete data ingestion pipeline:
    1. Creates a new Qdrant collection for storing embeddings
    2. Converts PDF document into text chunks
    3. Generates embeddings for each chunk using sentence-transformers
    4. Ingests embeddings and text into Qdrant for semantic search

    Note: This should be run ONCE to populate the database.
          Comment out after initial data ingestion to avoid duplicates.

    Steps:
        - create_qdrant_collection: Creates vector collection with proper dimensions
        - convert_pdf_to_chunks: Uses Docling to extract and chunk PDF content
        - get_embeddings_and_text: Generates vector embeddings for semantic search
        - ingest_vectors_in_qdrant: Stores vectors in Qdrant database

    Returns:
        None
    """
    create_qdrant_collection(COLLECTION_NAME)
    chunks = convert_pdf_to_chunks("HR-POLICIES-1-1-1.pdf")
    embeddings_and_text = get_embeddings_and_text(chunks)
    ingest_vectors_in_qdrant(COLLECTION_NAME, embeddings_and_text)


# Uncomment the line below to run data preparation (only needed once)
# prepare_data() 

# Create the Azure OpenAI agent with RAG capabilities
# The agent is configured with:
# - Specific instructions to only use retrieved information (no hallucinations)
# - Access to the Qdrant search tool for retrieving relevant HR policy chunks
# - Structured output format for answers with confidence scores
agent = client.create_agent(
    name="AgenticRagAgent",
    instructions="""You are a helpful assistant who is an expert in HR policies.
    You will never reply from your knowledge and will always use the tools to find the answer.
    If you dont find the answer you will simply say, "I am sorry, I could not find the answer."
    """,
    tools=[get_prepared_collection(COLLECTION_NAME)]
)


class AnswerAndConfidence(BaseModel):
    """
    Pydantic model for structured agent responses.

    This ensures the agent returns both an answer and a confidence score,
    providing transparency about the reliability of the response.

    Attributes:
        answer (str): The answer to the user's question about HR policies
        confidence (float): Confidence score between 0.0 and 1.0 indicating
                          how confident the agent is in the answer based on
                          the retrieved information
    """
    answer: str = Field(..., description="Answer to the question")
    confidence: float = Field(..., description="Confidence score of the answer")

async def main():
    """
    Main execution function for the Agentic RAG system.

    This async function demonstrates how to query the agent with a question
    about HR policies and receive a structured response with confidence score.

    The agent will:
    1. Receive the user's question
    2. Use the search tool to find relevant HR policy chunks from Qdrant
    3. Generate an answer based ONLY on retrieved information
    4. Return a structured response with answer and confidence score

    Returns:
        AnswerAndConfidence: Structured response containing the answer and confidence

    Example:
        response = await agent.run(
            "What is the remote work policy?",
            response_format=AnswerAndConfidence
        )
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
    """
    print("In Main")

    # Query the agent with a question about HR policies
    # The response_format ensures we get structured output matching AnswerAndConfidence model
    response = await agent.run(
        "What is the pre employment health check up policy?",
        response_format=AnswerAndConfidence
    )

    print(response)


# Entry point: Run the async main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())