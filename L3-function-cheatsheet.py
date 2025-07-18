#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# OPENAI FUNCTION CALLING IN LANGCHAIN QUICK REFERENCE
# =============================================================================
# Learn how to use OpenAI Function Calling with Pydantic models in LangChain
# This cheatsheet covers function definitions, binding, and usage patterns
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
# SECTION 1: PYDANTIC BASICS
# =============================================================================
# Understanding Pydantic data classes and validation
# =============================================================================

from typing import List
from pydantic import BaseModel, Field

def pydantic_basics_example():
    """Demonstrate basic Pydantic concepts"""
    
    # Standard Python class (no validation)
    class User:
        def __init__(self, name: str, age: int, email: str):
            self.name = name
            self.age = age
            self.email = email
    
    # This works but doesn't validate types
    foo = User(name="Joe", age=32, email="joe@gmail.com")
    print(f"Standard Python class: {foo.name}, {foo.age}")
    
    # This also works but age is a string, not int
    foo = User(name="Joe", age="bar", email="joe@gmail.com")  # type: ignore
    print(f"Invalid type accepted: {foo.age} (type: {type(foo.age)})")
    
    # Pydantic class with validation
    class PydanticUser(BaseModel):
        name: str
        age: int
        email: str
    
    # This works correctly
    foo_p = PydanticUser(name="Jane", age=32, email="jane@gmail.com")
    print(f"Pydantic class: {foo_p.name}, {foo_p.age}")
    
    # This would raise a validation error
    try:
        foo_p = PydanticUser(name="Jane", age="bar", email="jane@gmail.com")  # type: ignore
    except Exception as e:
        print(f"Pydantic validation error: {e}")
    
    # Nested Pydantic models
    class Class(BaseModel):
        students: List[PydanticUser]
    
    obj = Class(
        students=[PydanticUser(name="Jane", age=32, email="jane@gmail.com")]
    )
    print(f"Nested model: {obj}")
    
    return obj

# =============================================================================
# SECTION 2: PYDANTIC TO OPENAI FUNCTION CONVERSION
# =============================================================================
# Converting Pydantic models to OpenAI function definitions
# =============================================================================

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

def function_conversion_example():
    """Demonstrate converting Pydantic models to OpenAI functions"""
    
    # Correct way: Include docstring and Field descriptions
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    weather_function = convert_pydantic_to_openai_function(WeatherSearch)
    print(f"Weather function: {weather_function}")
    
    # Wrong way: Missing docstring (will fail)
    class WeatherSearch1(BaseModel):
        airport_code: str = Field(description="airport code to get weather for")
    
    try:
        convert_pydantic_to_openai_function(WeatherSearch1)
    except Exception as e:
        print(f"Missing docstring error: {e}")
    
    # Wrong way: Missing Field description (will fail)
    class WeatherSearch2(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str
    
    try:
        convert_pydantic_to_openai_function(WeatherSearch2)
    except Exception as e:
        print(f"Missing Field description error: {e}")
    
    return weather_function

# =============================================================================
# SECTION 3: BASIC FUNCTION CALLING
# =============================================================================
# Using functions with ChatOpenAI models
# =============================================================================

from langchain_openai import ChatOpenAI

def basic_function_calling():
    """Demonstrate basic function calling"""
    
    # Create the function
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    weather_function = convert_pydantic_to_openai_function(WeatherSearch)
    
    # Create model
    model = ChatOpenAI()
    
    # Method 1: Pass functions directly to invoke
    response = model.invoke("what is the weather in SF today?", functions=[weather_function])
    print(f"Direct function call: {response}")
    
    # Method 2: Bind functions to model
    model_with_function = model.bind(functions=[weather_function])
    response = model_with_function.invoke("what is the weather in sf?")
    print(f"Bound function call: {response}")
    
    return response

# =============================================================================
# SECTION 4: FORCED FUNCTION CALLING
# =============================================================================
# Forcing the model to use specific functions
# =============================================================================

def forced_function_calling():
    """Demonstrate forcing the model to use specific functions"""
    
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    weather_function = convert_pydantic_to_openai_function(WeatherSearch)
    model = ChatOpenAI()
    
    # Force the model to use the WeatherSearch function
    model_with_forced_function = model.bind(
        functions=[weather_function], 
        function_call={"name": "WeatherSearch"}
    )
    
    # This will always call the weather function
    response = model_with_forced_function.invoke("what is the weather in sf?")
    print(f"Forced weather function: {response}")
    
    # Even this will call the weather function
    response = model_with_forced_function.invoke("hi!")
    print(f"Forced function on greeting: {response}")
    
    return response

# =============================================================================
# SECTION 5: FUNCTIONS IN CHAINS
# =============================================================================
# Using function-bound models in LangChain chains
# =============================================================================

from langchain.prompts import ChatPromptTemplate

def functions_in_chains():
    """Demonstrate using functions in chains"""
    
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    weather_function = convert_pydantic_to_openai_function(WeatherSearch)
    model = ChatOpenAI()
    
    # Bind function to model
    model_with_function = model.bind(functions=[weather_function])
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", "{input}")
    ])
    
    # Create chain with function-bound model
    chain = prompt | model_with_function
    
    # Use the chain
    response = chain.invoke({"input": "what is the weather in sf?"})
    print(f"Chain with function: {response}")
    
    return response

# =============================================================================
# SECTION 6: MULTIPLE FUNCTIONS
# =============================================================================
# Using multiple functions and letting the LLM choose
# =============================================================================

def multiple_functions_example():
    """Demonstrate using multiple functions"""
    
    # Define multiple functions
    class WeatherSearch(BaseModel):
        """Call this with an airport code to get the weather at that airport"""
        airport_code: str = Field(description="airport code to get weather for")
    
    class ArtistSearch(BaseModel):
        """Call this to get the names of songs by a particular artist"""
        artist_name: str = Field(description="name of artist to look up")
        n: int = Field(description="number of results")
    
    # Convert to OpenAI functions
    functions = [
        convert_pydantic_to_openai_function(WeatherSearch),
        convert_pydantic_to_openai_function(ArtistSearch),
    ]
    
    # Bind multiple functions to model
    model = ChatOpenAI()
    model_with_functions = model.bind(functions=functions)
    
    # Let the LLM choose which function to use
    weather_response = model_with_functions.invoke("what is the weather in sf?")
    print(f"Weather query: {weather_response}")
    
    artist_response = model_with_functions.invoke("what are three songs by taylor swift?")
    print(f"Artist query: {artist_response}")
    
    # General conversation (no function needed)
    general_response = model_with_functions.invoke("hi!")
    print(f"General conversation: {general_response}")
    
    return weather_response, artist_response, general_response

# =============================================================================
# SECTION 7: ADVANCED PATTERNS
# =============================================================================
# Advanced function calling patterns for production use
# =============================================================================

def advanced_patterns():
    """Demonstrate advanced function calling patterns"""
    
    # Pattern 1: Function with complex validation
    class ComplexSearch(BaseModel):
        """Search for items with complex criteria"""
        query: str = Field(description="search query")
        filters: Dict[str, Any] = Field(
            description="optional filters",
            default_factory=dict
        )
        limit: int = Field(
            description="maximum number of results",
            ge=1,
            le=100,
            default=10
        )
    
    # Pattern 2: Function with nested models
    class Address(BaseModel):
        street: str = Field(description="street address")
        city: str = Field(description="city name")
        country: str = Field(description="country name")
    
    class UserProfile(BaseModel):
        """Create or update a user profile"""
        name: str = Field(description="user's full name")
        email: str = Field(description="user's email address")
        age: int = Field(description="user's age", ge=0, le=150)
        address: Address = Field(description="user's address")
        interests: List[str] = Field(
            description="list of user interests",
            default_factory=list
        )
    
    print("Advanced patterns available:")
    print("- Complex validation with Field constraints")
    print("- Nested Pydantic models")
    print("- Default values and optional fields")
    print("- List and dictionary fields")
    
    return ComplexSearch, UserProfile

# =============================================================================
# USAGE EXAMPLES & MAIN EXECUTION
# =============================================================================
# Run examples and demonstrate function calling functionality
# =============================================================================

if __name__ == "__main__":
    print("=== OpenAI Function Calling Examples ===\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'\n")
    
    try:
        # Example 1: Pydantic basics
        print("1. Pydantic Basics Example:")
        pydantic_basics_example()
        print()
        
        # Example 2: Function conversion
        print("2. Function Conversion Example:")
        function_conversion_example()
        print()
        
        # Example 3: Basic function calling
        print("3. Basic Function Calling Example:")
        basic_function_calling()
        print()
        
        # Example 4: Forced function calling
        print("4. Forced Function Calling Example:")
        forced_function_calling()
        print()
        
        # Example 5: Functions in chains
        print("5. Functions in Chains Example:")
        functions_in_chains()
        print()
        
        # Example 6: Multiple functions
        print("6. Multiple Functions Example:")
        multiple_functions_example()
        print()
        
        # Example 7: Advanced patterns
        print("7. Advanced Patterns:")
        advanced_patterns()
        print()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")
    
    print("=== Function Calling Cheatsheet Complete ===")
    print("Key Function Calling Concepts:")
    print("- Use Pydantic models with docstrings and Field descriptions")
    print("- Convert models to functions with convert_pydantic_to_openai_function")
    print("- Bind functions to models with .bind()")
    print("- Force function calls with function_call parameter")
    print("- Use multiple functions and let LLM choose")
    print("- Integrate functions into LangChain chains")

