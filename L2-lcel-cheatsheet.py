#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# LANGCHAIN EXPRESSION LANGUAGE (LCEL) QUICK REFERENCE
# =============================================================================
# LCEL is a declarative way to compose chains and other LangChain components
# This cheatsheet covers LCEL syntax and patterns for building complex workflows
# =============================================================================

import os
import json
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
# SECTION 1: BASIC LCEL SYNTAX
# =============================================================================
# Understanding the pipe operator and basic chain composition
# =============================================================================

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# Simple Chain using pipe operator
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# The magic of LCEL: pipe operator for composition
chain = prompt | model | output_parser

def simple_chain_example():
    """Demonstrate basic LCEL chain"""
    result = chain.invoke({"topic": "bears"})
    print(f"Simple chain result: {result}")
    return result

# =============================================================================
# SECTION 2: COMPLEX CHAINS WITH RETRIEVAL
# =============================================================================
# Using RunnableMap to supply user-provided inputs to the prompt
# =============================================================================

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.runnable import RunnableMap

# Create a simple vector store
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

def retrieval_example():
    """Demonstrate retrieval with LCEL"""
    # Test retrieval
    docs = retriever.get_relevant_documents("where did harrison work?")
    print(f"Retrieved documents: {docs}")
    
    # Create template for Q&A
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build complex chain with RunnableMap
    def get_context(input_dict: Dict[str, Any]) -> List:
        return retriever.get_relevant_documents(input_dict.get("question", ""))
    
    def get_question(input_dict: Dict[str, Any]) -> str:
        return input_dict.get("question", "")
    
    chain = RunnableMap({
        "context": get_context,
        "question": get_question
    }) | prompt | model | output_parser
    
    result = chain.invoke({"question": "where did harrison work?"})
    print(f"Q&A result: {result}")
    return result

# =============================================================================
# SECTION 3: FUNCTION BINDING
# =============================================================================
# Binding OpenAI Functions to models for structured outputs
# =============================================================================

def function_binding_example():
    """Demonstrate OpenAI function binding with LCEL"""
    
    # Define functions for the model to use
    functions = [
        {
            "name": "weather_search",
            "description": "Search for weather given an airport code",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "The airport code to get the weather for"
                    },
                },
                "required": ["airport_code"]
            }
        }
    ]
    
    # Create prompt and bind functions to model
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
    model_with_functions = ChatOpenAI(temperature=0).bind(functions=functions)
    
    # Create runnable chain
    runnable = prompt | model_with_functions
    
    result = runnable.invoke({"input": "what is the weather in sf"})
    print(f"Function binding result: {result}")
    return result

def multiple_functions_example():
    """Demonstrate multiple function binding"""
    
    functions = [
        {
            "name": "weather_search",
            "description": "Search for weather given an airport code",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "The airport code to get the weather for"
                    },
                },
                "required": ["airport_code"]
            }
        },
        {
            "name": "sports_search",
            "description": "Search for news of recent sport events",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {
                        "type": "string",
                        "description": "The sports team to search for"
                    },
                },
                "required": ["team_name"]
            }
        }
    ]
    
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
    model_with_functions = ChatOpenAI(temperature=0).bind(functions=functions)
    runnable = prompt | model_with_functions
    
    result = runnable.invoke({"input": "how did the patriots do yesterday?"})
    print(f"Multiple functions result: {result}")
    return result

# =============================================================================
# SECTION 4: FALLBACKS
# =============================================================================
# Implementing fallback mechanisms for robust chains
# =============================================================================

from langchain_openai import OpenAI

def fallback_example():
    """Demonstrate fallback mechanisms"""
    
    # Create a simple model that might fail
    simple_model = OpenAI(
        temperature=0, 
        max_tokens=1000, 
        model="gpt-3.5-turbo-instruct"
    )
    simple_chain = simple_model | json.loads
    
    # Create a more robust model as fallback
    robust_model = ChatOpenAI(temperature=0)
    robust_chain = robust_model | StrOutputParser() | json.loads
    
    # Challenge that might cause the simple chain to fail
    challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
    
    # Create fallback chain
    final_chain = simple_chain.with_fallbacks([robust_chain])
    
    try:
        result = final_chain.invoke(challenge)
        print(f"Fallback chain result: {result}")
        return result
    except Exception as e:
        print(f"Fallback chain failed: {e}")
        return None

# =============================================================================
# SECTION 5: INTERFACE METHODS
# =============================================================================
# Different ways to invoke LCEL chains
# =============================================================================

def interface_examples():
    """Demonstrate different interface methods for LCEL chains"""
    
    # Create a simple chain
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    model = ChatOpenAI()
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    
    # 1. Single invocation
    print("1. Single invocation:")
    result = chain.invoke({"topic": "bears"})
    print(f"   Result: {result}")
    
    # 2. Batch processing
    print("\n2. Batch processing:")
    results = chain.batch([{"topic": "bears"}, {"topic": "frogs"}])
    print(f"   Results: {results}")
    
    # 3. Streaming
    print("\n3. Streaming:")
    for chunk in chain.stream({"topic": "bears"}):
        print(f"   Chunk: {chunk}")
    
    # 4. Async invocation (commented out as it requires async context)
    # print("\n4. Async invocation:")
    # response = await chain.ainvoke({"topic": "bears"})
    # print(f"   Async result: {response}")
    
    return results

# =============================================================================
# SECTION 6: ADVANCED PATTERNS
# =============================================================================
# Complex LCEL patterns for production use
# =============================================================================

def advanced_patterns():
    """Demonstrate advanced LCEL patterns"""
    
    # Pattern 1: Conditional routing
    from langchain.schema.runnable import RunnableLambda
    
    def route_by_topic(input_dict):
        topic = input_dict.get("topic", "").lower()
        if "weather" in topic:
            return {"route": "weather", "query": input_dict["topic"]}
        elif "sports" in topic:
            return {"route": "sports", "query": input_dict["topic"]}
        else:
            return {"route": "general", "query": input_dict["topic"]}
    
    router = RunnableLambda(route_by_topic)
    
    # Pattern 2: Parallel processing
    from langchain.schema.runnable import RunnableParallel
    
    parallel_chain = RunnableParallel({
        "joke": chain,
        "explanation": prompt | model | StrOutputParser()
    })
    
    print("Advanced patterns available:")
    print("- Conditional routing with RunnableLambda")
    print("- Parallel processing with RunnableParallel")
    print("- Custom output parsing")
    
    return router, parallel_chain

# =============================================================================
# USAGE EXAMPLES & MAIN EXECUTION
# =============================================================================
# Run examples and demonstrate LCEL functionality
# =============================================================================

if __name__ == "__main__":
    print("=== LangChain Expression Language (LCEL) Examples ===\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'\n")
    
    try:
        # Example 1: Simple chain
        print("1. Simple Chain Example:")
        simple_chain_example()
        print()
        
        # Example 2: Retrieval chain
        print("2. Retrieval Chain Example:")
        retrieval_example()
        print()
        
        # Example 3: Function binding
        print("3. Function Binding Example:")
        function_binding_example()
        print()
        
        # Example 4: Multiple functions
        print("4. Multiple Functions Example:")
        multiple_functions_example()
        print()
        
        # Example 5: Fallbacks
        print("5. Fallback Example:")
        fallback_example()
        print()
        
        # Example 6: Interface methods
        print("6. Interface Methods Example:")
        interface_examples()
        print()
        
        # Example 7: Advanced patterns
        print("7. Advanced Patterns:")
        advanced_patterns()
        print()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")
    
    print("=== LCEL Cheatsheet Complete ===")
    print("Key LCEL Concepts:")
    print("- Use | operator for chain composition")
    print("- RunnableMap for input transformation")
    print("- .bind() for function binding")
    print("- .with_fallbacks() for error handling")
    print("- .batch(), .stream(), .ainvoke() for different interfaces")

