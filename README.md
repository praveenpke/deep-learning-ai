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

## 💡 Example Use Cases

### 1️⃣ Simple Chat with OpenAI
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

Run the cheatsheet to see examples in action:

```bash
python langchain_cheatsheet.py
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