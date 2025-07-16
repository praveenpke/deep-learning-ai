
# =============================================================================
# LANGCHAIN QUICK REFERENCE (PYTHON VERSION)
# =============================================================================
# Updated for LangChain 0.1+ with working examples
# 
# This cheatsheet covers all major LangChain components with practical examples
# =============================================================================

import os
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables directly in your shell")
except FileNotFoundError:
    print("No .env file found. Using system environment variables")

# =============================================================================
# SECTION 1: MODEL, PROMPT, OUTPUT PARSER
# =============================================================================
# Core components for working with LLMs, creating prompts, and parsing responses
# =============================================================================

# Model
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize with API key (set your OpenAI API key in environment)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Simple chat example
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about programming")
]
response = llm.invoke(messages)
print(f"Response: {response.content}")

# Prompt Template
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Text prompt template
text_prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
formatted_prompt = text_prompt.format(topic="artificial intelligence")
print(f"Formatted prompt: {formatted_prompt}")

# Chat prompt template
system_template = "You are a helpful assistant that specializes in {domain}."
human_template = "Explain {concept} in simple terms."

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt
])

# Output Parser (example for structured output)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import Optional

# Using ResponseSchema
response_schemas = [
    ResponseSchema(name="name", description="Person name", type="string"),
    ResponseSchema(name="age", description="Person age", type="integer"),
    ResponseSchema(name="occupation", description="Person occupation", type="string")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Using Pydantic model (recommended approach)
class Person(BaseModel):
    name: str = Field(description="Person name")
    age: int = Field(description="Person age")
    occupation: str = Field(description="Person occupation")
    hobbies: Optional[List[str]] = Field(description="List of hobbies", default=[])

# =============================================================================
# SECTION 2: MEMORY
# =============================================================================
# Managing conversation history and context across interactions
# =============================================================================

# ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Example usage with memory
from langchain.chains import ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ConversationSummaryMemory
from langchain.memory import ConversationSummaryMemory
summary_memory = ConversationSummaryMemory(
    llm=llm
)

# =============================================================================
# SECTION 3: CHAINS
# =============================================================================
# Combining multiple components to create complex workflows
# =============================================================================

# LLMChain
from langchain.chains import LLMChain

llm_chain = LLMChain(
    llm=llm,
    prompt=text_prompt,
    verbose=True
)

# Example usage
result = llm_chain.run(topic="machine learning")
print(f"Chain result: {result}")

# SequentialChain
from langchain.chains import SequentialChain

# Create individual chains
title_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("Write a catchy title for a blog post about {topic}"),
    output_key="title"
)

outline_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("Create an outline for a blog post titled: {title}"),
    output_key="outline"
)

# Combine chains
seq_chain = SequentialChain(
    chains=[title_chain, outline_chain],
    input_variables=["topic"],
    output_variables=["title", "outline"],
    verbose=True
)

# =============================================================================
# SECTION 4: QnA OVER DOCUMENTS
# =============================================================================
# Building question-answering systems over your own documents
# =============================================================================

# Vector Store Indexing and Querying
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load and split documents
def create_vectorstore_from_text(file_path: str):
    """Create a vector store from a text file"""
    try:
        # Load document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data...")
        # Create sample data for demonstration
        sample_text = """
        This is a sample document about artificial intelligence.
        AI has revolutionized many industries including healthcare, finance, and transportation.
        Machine learning is a subset of AI that focuses on algorithms that can learn from data.
        Deep learning uses neural networks with multiple layers to process complex patterns.
        """
        
        # Save sample text to file
        with open("sample_data.txt", "w") as f:
            f.write(sample_text)
        
        # Now load the sample file
        return create_vectorstore_from_text("sample_data.txt")

# Create QA chain
def create_qa_chain(vectorstore):
    """Create a question-answering chain"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

# Example usage (commented out to avoid errors if file doesn't exist)
# vectorstore = create_vectorstore_from_text("data.txt")
# qa_chain = create_qa_chain(vectorstore)
# response = qa_chain({"query": "What is machine learning?"})

# =============================================================================
# SECTION 5: EVALUATION
# =============================================================================
# Assessing the quality and accuracy of LLM responses
# =============================================================================

# Manual Evaluation (simplified approach)
def evaluate_qa_response(predicted: str, expected: str) -> Dict[str, Any]:
    """Simple evaluation function"""
    # Basic exact match
    exact_match = predicted.lower().strip() == expected.lower().strip()
    
    # Keyword matching
    predicted_words = set(predicted.lower().split())
    expected_words = set(expected.lower().split())
    keyword_overlap = len(predicted_words.intersection(expected_words)) / len(expected_words)
    
    return {
        "exact_match": exact_match,
        "keyword_overlap": keyword_overlap,
        "predicted": predicted,
        "expected": expected
    }

# Example evaluation
sample_evaluation = evaluate_qa_response(
    predicted="Paris is the capital of France",
    expected="Paris is the capital of France"
)
print(f"Evaluation result: {sample_evaluation}")

# =============================================================================
# SECTION 6: AGENTS
# =============================================================================
# Creating autonomous agents that can use tools and make decisions
# =============================================================================

# Built-in Tools
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool

@tool
def get_weather_data(location: str) -> str:
    """Fetch weather information for a given location"""
    # This is a mock implementation - in real usage, you'd call a weather API
    return f"The weather in {location} is sunny with a temperature of 22°C."

@tool
def calculate_math(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

# Create agent
def create_simple_agent():
    """Create a simple agent with custom tools"""
    tools = [get_weather_data, calculate_math]
    
    # Create prompt template for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools."),
        ("human", "{input}"),
        ("human", "Use tools if needed to answer the question.")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Example usage (commented out to avoid API calls)
# agent = create_simple_agent()
# result = agent.invoke({"input": "What is the weather like in Tokyo?"})

# =============================================================================
# SECTION 7: ADVANCED FEATURES
# =============================================================================
# Production-ready features for building robust applications
# =============================================================================

# Streaming
def stream_response():
    """Example of streaming responses"""
    streaming_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True
    )
    
    messages = [HumanMessage(content="Write a short story about a robot learning to paint.")]
    
    print("Streaming response:")
    for chunk in streaming_llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

# Error handling wrapper
def safe_llm_call(func, *args, **kwargs):
    """Wrapper to handle LLM API errors gracefully"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        return None

# =============================================================================
# USAGE EXAMPLES & MAIN EXECUTION
# =============================================================================
# Run examples and demonstrate functionality
# =============================================================================

if __name__ == "__main__":
    print("=== LangChain Cheatsheet Examples ===\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'\n")
    
    # Example 1: Simple chat
    print("1. Simple Chat Example:")
    try:
        response = safe_llm_call(llm.invoke, messages)
        if response:
            print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Chain usage
    print("2. Chain Example:")
    try:
        result = safe_llm_call(llm_chain.run, topic="space exploration")
        if result:
            print(f"Result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Evaluation
    print("3. Evaluation Example:")
    eval_result = evaluate_qa_response(
        "Machine learning is a subset of AI",
        "Machine learning is a subset of artificial intelligence"
    )
    print(f"Evaluation: {eval_result}\n")
    
    print("=== Cheatsheet Complete ===")
    print("Remember to:")
    print("- Set your OPENAI_API_KEY environment variable")
    print("- Install required packages: pip install langchain langchain-openai faiss-cpu wikipedia")
    print("- Handle API rate limits and errors in production code")
