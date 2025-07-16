# LangChain Cheatsheet

A comprehensive quick reference guide for LangChain with working examples and best practices.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key (choose one method):**

   **Option A: Using .env file (recommended for development):**
   ```bash
   cp .env.example .env
   # OR
   cp .env.sample .env
   # Edit .env file and add your actual API key
   ```

   **Option B: Using environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

## Features

The cheatsheet includes examples for:

### L1: Model, Prompt, Output Parser
- ChatOpenAI model initialization
- Prompt templates (text and chat)
- Output parsers with ResponseSchema and Pydantic models

### L2: Memory
- ConversationBufferMemory
- ConversationSummaryMemory
- Memory integration with chains

### L3: Chains
- LLMChain
- SequentialChain
- Chain composition and execution

### L4: QnA Over Documents
- Document loading and text splitting
- Vector store creation with FAISS
- Retrieval QA chains

### L5: Evaluation
- Custom evaluation functions
- Response quality assessment

### L6: Agents
- Custom tool creation
- Agent initialization and execution
- Tool integration

### L7: Advanced Features
- Streaming responses
- Error handling
- Production-ready patterns

## Usage

Run the cheatsheet to see examples in action:

```bash
python langchain_cheatsheet.py
```

## Key Improvements Made

1. **Fixed all linter errors** - Updated imports and parameter names for current LangChain version
2. **Added proper error handling** - Graceful handling of missing API keys and file errors
3. **Comprehensive examples** - Working code snippets for each LangChain component
4. **Production-ready patterns** - Best practices for real-world usage
5. **Documentation** - Clear comments and explanations throughout

## Notes

- The file includes sample data creation for demonstration purposes
- API calls are wrapped in error handling to prevent crashes
- Examples are designed to work with minimal setup
- All imports use the current LangChain package structure

## Dependencies

- `langchain` - Core LangChain functionality
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools and integrations
- `faiss-cpu` - Vector store for document retrieval
- `openai` - OpenAI API client
- `pydantic` - Data validation
- `python-dotenv` - Environment variable management (optional) 