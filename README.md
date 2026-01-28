# Agentic RAG for HR Policy Q&A

An intelligent question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions about HR policies. The system leverages Azure OpenAI agents, vector search with Qdrant, and document processing to provide accurate, grounded responses with confidence scores.

## 🎯 Purpose

This project demonstrates how to build an agentic RAG system that:
- **Processes HR policy documents** (PDFs) and converts them into searchable knowledge
- **Provides accurate answers** to employee questions about HR policies
- **Prevents hallucinations** by only answering from retrieved documents
- **Returns confidence scores** to indicate answer reliability
- **Uses Azure OpenAI agents** with tool-calling capabilities for intelligent retrieval

## 🏗️ Architecture

1. **Document Ingestion**: PDF documents are processed using Docling, chunked, and embedded using sentence-transformers
2. **Vector Storage**: Embeddings are stored in Qdrant vector database for semantic search
3. **Agent System**: Azure OpenAI agent acts as an HR expert with access to a search tool
4. **RAG Pipeline**: User queries trigger vector search, and the agent generates grounded answers

## 📋 Prerequisites

- **Python 3.12** (recommended) - Must be between 3.10 and 3.12 (inclusive)
  - ⚠️ **Important**: Use Python 3.12 for best compatibility
  - Minimum: Python 3.10
  - Maximum: Python 3.12
- Docker (for Qdrant)
- Azure OpenAI API access

## 🚀 Installation & Setup

### Step 1: Install UV Package Manager

UV is a fast Python package installer and resolver. Install it using pip:

```bash
pip install uv
```

**Alternative installation methods:**

<details>
<summary>Click to see platform-specific installers</summary>

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
</details>

### Step 2: Pull and Run Qdrant Docker Image

Qdrant is used as the vector database. Pull and run it using Docker:

**Pull the Qdrant image (same for all platforms):**
```bash
docker pull qdrant/qdrant
```

**macOS/Linux:**
```bash
# Run Qdrant container
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

**Windows (PowerShell):**
```powershell
# Run Qdrant container
docker run -p 6333:6333 -p 6334:6334 `
    -v ${PWD}/qdrant_storage:/qdrant/storage:z `
    qdrant/qdrant
```

**Windows (Command Prompt):**
```cmd
# Run Qdrant container
docker run -p 6333:6333 -p 6334:6334 ^
    -v %cd%/qdrant_storage:/qdrant/storage:z ^
    qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`

### Step 3: Install Project Dependencies

Use UV to sync and install all required libraries:

```bash
# Sync dependencies from pyproject.toml and uv.lock
uv sync
```

This will create a virtual environment (`.venv`) and install all dependencies specified in `pyproject.toml` with exact versions from `uv.lock`.

### Step 4: Activate the Virtual Environment

After running `uv sync`, activate the virtual environment:

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your terminal prompt indicating the environment is active.

**Key Dependencies:**
- `azure-ai-projects` - Azure OpenAI agent framework
- `qdrant-client` - Vector database client
- `docling` - PDF document processing
- `sentence-transformers` - Text embeddings
- `pydantic` - Data validation

### Step 5: Configure Environment Variables

Create a `.env` file or set environment variables for Azure OpenAI:

```bash
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

### Step 6: Prepare Your Data

Place your HR policy PDF documents in the project directory and update the file path in `utils.py` if needed.

### Step 7: Run the Application

Make sure your virtual environment is activated, then run:

```bash
# Run the main application
python main.py
```

## 📁 Project Structure

- `main.py` - Main application with Azure OpenAI agent setup
- `tools.py` - RAG tool for searching Qdrant vector database
- `utils.py` - Utility functions for PDF processing and embedding
- `requirements.txt` - Project dependencies

## 🔧 How It Works

1. **Ingestion**: HR policy PDFs are processed and stored as vector embeddings in Qdrant
2. **Query**: User asks a question about HR policies
3. **Retrieval**: Agent calls the search tool to find relevant document chunks
4. **Generation**: Agent generates an answer based only on retrieved content
5. **Response**: Returns answer with confidence score (0.0-1.0)

## 💡 Usage Example

The agent will answer questions like:
- "What is the company's remote work policy?"
- "How many vacation days do employees get?"
- "What is the process for requesting leave?"

If the information isn't found in the documents, the agent will indicate it cannot answer the question.

## 🛡️ Features

- ✅ Grounded responses (no hallucinations)
- ✅ Confidence scoring
- ✅ Semantic search with vector embeddings
- ✅ Structured output using Pydantic models
- ✅ Agent-based architecture with tool calling

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to fork, modify, and use this project as a template for your own RAG applications!

