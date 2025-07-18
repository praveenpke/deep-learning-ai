# 🦜🔗 LangChain Cheatsheet

[![LangChain](https://img.shields.io/badge/LangChain-Python-blue?logo=python)](https://python.langchain.com/) [![OpenAI](https://img.shields.io/badge/OpenAI-API-green?logo=openai)](https://platform.openai.com/)

A comprehensive quick reference guide for LangChain with working examples and best practices.

---

## 🚀 Features

- **LLM Chat**: Simple chat with OpenAI models
- **Prompt Engineering**: Templates for text and chat
- **Memory**: Conversation history and summarization
- **Chains**: Sequential and custom workflows
- **Document QnA**: Ask questions over your own files
- **Agents & Tools**: Custom tool integration
- **Evaluation**: Response quality checks
- **Streaming**: Real-time LLM output
- **LCEL**: LangChain Expression Language for declarative chain composition
- **Function Calling**: OpenAI Function Calling with Pydantic models
- **Tagging & Extraction**: Structured data extraction and text tagging

---

## 🛠️ Setup

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

---

## 📚 Cheatsheets

This repository contains four comprehensive cheatsheets designed to take you from basic LangChain usage to advanced patterns:

### 🔗 **L1-langchain_cheatsheet.py** - Core LangChain Components

**Why we use it:** This is your foundation - it covers all the essential LangChain building blocks you need to understand before moving to advanced patterns.

**When we use it:**
- Learning LangChain for the first time
- Building simple chatbots and Q&A systems
- Creating basic document processing workflows
- Implementing conversation memory
- Setting up agents with custom tools
- Evaluating LLM responses

**What's included:** Models, prompts, memory, chains, agents, document Q&A, evaluation, streaming, and advanced features.

---

### ⚡ **L2-lcel-cheatsheet.py** - LangChain Expression Language (LCEL)

**Why we use it:** LCEL provides a declarative, composable way to build complex chains using the pipe operator (`|`), making your code more readable and maintainable.

**When we use it:**
- Building complex multi-step workflows
- Creating reusable chain components
- Implementing fallback mechanisms
- Running operations in parallel
- Binding functions to chains
- Creating custom output parsers

**What's included:** Chain composition, function binding, fallbacks, parallel processing, and custom components.

---

### 🎯 **L3-function-cheatsheet.py** - OpenAI Function Calling

**Why we use it:** Function calling allows you to define structured schemas that LLMs can use to return data in specific formats, enabling more reliable and structured interactions.

**When we use it:**
- Extracting structured data from text
- Building APIs that need consistent output formats
- Creating tools that LLMs can call
- Implementing data validation
- Building agents with specific capabilities
- Converting between different data formats

**What's included:** Pydantic models, function conversion, validation, binding, and advanced function calling patterns.

---

### 📊 **L4-tagging-and-extraction.py** - Tagging and Extraction

**Why we use it:** This specialized pattern helps you extract specific information from text and tag content with metadata, making it perfect for content analysis and data processing.

**When we use it:**
- Analyzing sentiment in customer feedback
- Extracting entities from documents
- Categorizing content automatically
- Processing large batches of text
- Building content moderation systems
- Creating structured datasets from unstructured text

**What's included:** Text tagging, entity extraction, batch processing, document analysis, and structured data extraction.

---

## 💡 Example Use Cases

### 1️⃣ Simple Chat with OpenAI

**Why we use this:** The foundation of any LLM application - direct interaction with AI models to generate responses.

**When we use this:**
- Building chatbots and conversational AI
- Creating content generation systems
- Implementing simple Q&A systems
- Testing model responses and capabilities
- Building applications that need direct AI interaction

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about AI.")
]
response = llm.invoke(messages)
print(response.content)
```

---

### 2️⃣ QnA Over Your Documents

**Why we use this:** Allows AI to answer questions based on your specific knowledge base by retrieving relevant information from documents.

**When we use this:**
- Building knowledge base chatbots for companies
- Creating customer support systems with product documentation
- Implementing research assistants that can search through papers
- Building internal company Q&A systems
- Creating educational platforms with textbook Q&A capabilities

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load and split your document
loader = TextLoader("data.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

response = qa_chain({"query": "What is the main topic?"})
print(response["result"])
```

---

### 3️⃣ Custom Agent with Tools

**Why we use this:** Creates autonomous AI agents that can use external tools and make decisions about which tools to use for different tasks.

**When we use this:**
- Building AI assistants that can search the web
- Creating agents that can perform calculations
- Implementing AI that can interact with databases
- Building autonomous research assistants
- Creating AI systems that can use multiple external services

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate

@tool
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

tools = [get_weather]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What's the weather in Paris?"})
print(result["output"])
```

---

### 4️⃣ LCEL Chain Composition

**Why we use this:** LCEL provides a declarative, composable way to build complex chains using the pipe operator, making code more readable and maintainable.

**When we use this:**
- Building complex multi-step workflows
- Creating reusable chain components
- Implementing fallback mechanisms
- Running operations in parallel
- Binding functions to chains
- Creating custom output parsers

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# LCEL makes chain composition simple with the pipe operator
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# The magic of LCEL: pipe operator for composition
chain = prompt | model | output_parser

result = chain.invoke({"topic": "programming"})
print(result)
```

---

### 5️⃣ OpenAI Function Calling

**Why we use this:** Function calling allows you to define structured schemas that LLMs can use to return data in specific formats, enabling more reliable and structured interactions.

**When we use this:**
- Extracting structured data from text
- Building APIs that need consistent output formats
- Creating tools that LLMs can call
- Implementing data validation
- Building agents with specific capabilities
- Converting between different data formats

```python
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI

# Define a function with Pydantic
class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

# Convert to OpenAI function
weather_function = convert_pydantic_to_openai_function(WeatherSearch)

# Bind to model
model = ChatOpenAI()
model_with_function = model.bind(functions=[weather_function])

# Use the function
response = model_with_function.invoke("what is the weather in SF?")
print(response)
```

---

### 6️⃣ Text Tagging and Extraction

**Why we use this:** This specialized pattern helps you extract specific information from text and tag content with metadata, making it perfect for content analysis and data processing.

**When we use this:**
- Analyzing sentiment in customer feedback
- Extracting entities from documents
- Categorizing content automatically
- Processing large batches of text
- Building content moderation systems
- Creating structured datasets from unstructured text

```python
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI

# Define tagging schema
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text")
    language: str = Field(description="language of text")

# Convert to function and use
tagging_function = convert_pydantic_to_openai_function(Tagging)
model = ChatOpenAI()
model_with_function = model.bind(functions=[tagging_function])

# Tag text
response = model_with_function.invoke("I love this product!")
print(response)
```

---

## 🌈 Visual Guide

> **Section Highlights:**
>
> - `SECTION 1:` Model, Prompt, Output Parser
> - `SECTION 2:` Memory
> - `SECTION 3:` Chains
> - `SECTION 4:` QnA Over Documents
> - `SECTION 5:` Evaluation
> - `SECTION 6:` Agents
> - `SECTION 7:` Advanced Features

Each section in `langchain_cheatsheet.py` is clearly marked with big comment headings for easy navigation.

---

## 📄 Usage

Run the cheatsheets to see examples in action:

```bash
# Core LangChain components
python langchain_cheatsheet.py

# LCEL patterns and advanced composition
python L2-lcel-cheatsheet.py

# OpenAI Function Calling with Pydantic
python L3-function-cheatsheet.py

# Tagging and Extraction
python L4-tagging-and-extraction.py
```

---

## 📦 Dependencies

- `langchain` - Core LangChain functionality
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools and integrations
- `faiss-cpu` - Vector store for document retrieval
- `openai` - OpenAI API client
- `pydantic` - Data validation
- `python-dotenv` - Environment variable management (optional)

---

## 📝 Notes

- The file includes sample data creation for demonstration purposes
- API calls are wrapped in error handling to prevent crashes
- Examples are designed to work with minimal setup
- All imports use the current LangChain package structure

---

## ⭐️ Star this repo if you find it useful!

---

Made with ❤️ for the AI developer community. 