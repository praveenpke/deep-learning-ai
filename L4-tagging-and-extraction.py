# =============================================================================
# TAGGING AND EXTRACTION USING OPENAI FUNCTIONS QUICK REFERENCE
# =============================================================================
# Learn how to extract structured data and tag text using OpenAI functions
# This cheatsheet covers tagging, extraction, and processing large documents
# =============================================================================

import os
from typing import List, Optional, Dict, Any

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
# SECTION 1: BASIC TAGGING
# =============================================================================
# Tagging text with specific attributes like sentiment and language
# =============================================================================

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

def basic_tagging_example():
    """Demonstrate basic text tagging"""
    
    # Define tagging schema
    class Tagging(BaseModel):
        """Tag the piece of text with particular info."""
        sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
        language: str = Field(description="language of text (should be ISO 639-1 code)")
    
    # Convert to OpenAI function
    tagging_function = convert_pydantic_to_openai_function(Tagging)
    print(f"Tagging function: {tagging_function}")
    
    # Create model and chain
    model = ChatOpenAI(temperature=0)
    tagging_functions = [tagging_function]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ])
    
    # Bind functions to model
    model_with_functions = model.bind(
        functions=tagging_functions,
        function_call={"name": "Tagging"}
    )
    
    # Create tagging chain
    tagging_chain = prompt | model_with_functions
    
    # Test with different texts
    result1 = tagging_chain.invoke({"input": "I love langchain"})
    print(f"Positive text: {result1}")
    
    result2 = tagging_chain.invoke({"input": "non mi piace questo cibo"})
    print(f"Italian text: {result2}")
    
    # Add JSON parser for cleaner output
    tagging_chain_with_parser = prompt | model_with_functions | JsonOutputFunctionsParser()
    result3 = tagging_chain_with_parser.invoke({"input": "non mi piace questo cibo"})
    print(f"Parsed result: {result3}")
    
    return result1, result2, result3

# =============================================================================
# SECTION 2: BASIC EXTRACTION
# =============================================================================
# Extracting structured information from text
# =============================================================================

def basic_extraction_example():
    """Demonstrate basic information extraction"""
    
    # Define extraction schemas
    class Person(BaseModel):
        """Information about a person."""
        name: str = Field(description="person's name")
        age: Optional[int] = Field(description="person's age")
    
    class Information(BaseModel):
        """Information to extract."""
        people: List[Person] = Field(description="List of info about people")
    
    # Convert to OpenAI function
    extraction_function = convert_pydantic_to_openai_function(Information)
    extraction_functions = [extraction_function]
    
    # Create model and chain
    model = ChatOpenAI(temperature=0)
    extraction_model = model.bind(
        functions=extraction_functions, 
        function_call={"name": "Information"}
    )
    
    # Test extraction
    result = extraction_model.invoke("Joe is 30, his mom is Martha")
    print(f"Basic extraction: {result}")
    
    # Create extraction chain with prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
        ("human", "{input}")
    ])
    
    extraction_chain = prompt | extraction_model
    result2 = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
    print(f"Chain extraction: {result2}")
    
    # Add JSON parser
    from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
    extraction_chain_with_parser = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
    result3 = extraction_chain_with_parser.invoke({"input": "Joe is 30, his mom is Martha"})
    print(f"Parsed extraction: {result3}")
    
    return result, result2, result3

# =============================================================================
# SECTION 3: DOCUMENT PROCESSING
# =============================================================================
# Processing larger documents with tagging and extraction
# =============================================================================

def document_processing_example():
    """Demonstrate processing larger documents"""
    
    # Define overview tagging schema
    class Overview(BaseModel):
        """Overview of a section of text."""
        summary: str = Field(description="Provide a concise summary of the content.")
        language: str = Field(description="Provide the language that the content is written in.")
        keywords: str = Field(description="Provide keywords related to the content.")
    
    # Create overview tagging chain
    model = ChatOpenAI(temperature=0)
    overview_tagging_function = [convert_pydantic_to_openai_function(Overview)]
    tagging_model = model.bind(
        functions=overview_tagging_function,
        function_call={"name": "Overview"}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Think carefully, and then tag the text as instructed"),
        ("user", "{input}")
    ])
    
    tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()
    
    # Example with sample text (instead of web scraping)
    sample_text = """
    Artificial Intelligence has revolutionized many industries. Machine learning algorithms 
    can now process vast amounts of data to identify patterns and make predictions. 
    Deep learning, a subset of machine learning, uses neural networks with multiple layers 
    to solve complex problems in computer vision, natural language processing, and more.
    """
    
    result = tagging_chain.invoke({"input": sample_text})
    print(f"Document overview: {result}")
    
    return result

# =============================================================================
# SECTION 4: PAPER EXTRACTION
# =============================================================================
# Extracting academic papers and references from text
# =============================================================================

def paper_extraction_example():
    """Demonstrate extracting papers and references"""
    
    # Define paper extraction schemas
    class Paper(BaseModel):
        """Information about papers mentioned."""
        title: str = Field(description="title of the paper")
        author: Optional[str] = Field(description="author of the paper")
    
    class Info(BaseModel):
        """Information to extract"""
        papers: List[Paper] = Field(description="List of papers mentioned")
    
    # Create extraction chain
    model = ChatOpenAI(temperature=0)
    paper_extraction_function = [convert_pydantic_to_openai_function(Info)]
    extraction_model = model.bind(
        functions=paper_extraction_function, 
        function_call={"name": "Info"}
    )
    
    from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
    
    # Create specialized prompt for paper extraction
    template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author. 

    Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

    Do not make up or guess ANY extra information. Only extract what exactly is in the text."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{input}")
    ])
    
    extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
    
    # Test with sample text containing paper references
    sample_text = """
    Recent advances in machine learning have been driven by several key papers. 
    "Attention Is All You Need" by Vaswani et al. introduced the transformer architecture.
    "BERT: Pre-training of Deep Bidirectional Transformers" by Devlin et al. 
    revolutionized natural language processing.
    """
    
    result = extraction_chain.invoke({"input": sample_text})
    print(f"Paper extraction: {result}")
    
    # Test with text that has no papers
    result2 = extraction_chain.invoke({"input": "hi"})
    print(f"No papers text: {result2}")
    
    return result, result2

# =============================================================================
# SECTION 5: BATCH PROCESSING
# =============================================================================
# Processing large documents by splitting and processing chunks
# =============================================================================

def batch_processing_example():
    """Demonstrate batch processing of large documents"""
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema.runnable import RunnableLambda
    
    # Define the extraction schema (reusing from previous section)
    class Paper(BaseModel):
        """Information about papers mentioned."""
        title: str = Field(description="title of the paper")
        author: Optional[str] = Field(description="author of the paper")
    
    class Info(BaseModel):
        """Information to extract"""
        papers: List[Paper] = Field(description="List of papers mentioned")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
    
    # Create extraction chain
    model = ChatOpenAI(temperature=0)
    paper_extraction_function = [convert_pydantic_to_openai_function(Info)]
    extraction_model = model.bind(
        functions=paper_extraction_function, 
        function_call={"name": "Info"}
    )
    
    template = """Extract all papers mentioned in this text. If no papers are mentioned, return an empty list."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", "{input}")
    ])
    
    from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
    extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
    
    # Helper function to flatten results
    def flatten(matrix):
        """Flatten a list of lists"""
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list
    
    # Create preprocessing step
    def split_and_prepare(text: str) -> List[Dict[str, str]]:
        """Split text and prepare for extraction"""
        return [{"input": doc} for doc in text_splitter.split_text(text)]
    
    prep = RunnableLambda(split_and_prepare)
    
    # Create batch processing chain
    chain = prep | extraction_chain.map() | flatten
    
    # Test with sample large text
    sample_large_text = """
    The field of artificial intelligence has seen remarkable progress in recent years.
    "Attention Is All You Need" by Vaswani et al. introduced transformers.
    "BERT: Pre-training of Deep Bidirectional Transformers" by Devlin et al. 
    showed the power of pre-trained language models.
    
    In computer vision, "ImageNet Classification with Deep Convolutional Neural Networks" 
    by Krizhevsky et al. demonstrated the effectiveness of deep learning.
    "ResNet" by He et al. introduced residual connections that enabled training of very deep networks.
    """
    
    result = chain.invoke(sample_large_text)
    print(f"Batch processing result: {result}")
    
    return result

# =============================================================================
# SECTION 6: ADVANCED EXTRACTION PATTERNS
# =============================================================================
# Advanced patterns for complex extraction scenarios
# =============================================================================

def advanced_extraction_patterns():
    """Demonstrate advanced extraction patterns"""
    
    # Pattern 1: Multi-level extraction
    class Address(BaseModel):
        street: str = Field(description="street address")
        city: str = Field(description="city name")
        country: str = Field(description="country name")
    
    class Company(BaseModel):
        name: str = Field(description="company name")
        industry: str = Field(description="industry sector")
        founded: Optional[int] = Field(description="founding year")
    
    class Contact(BaseModel):
        name: str = Field(description="contact person name")
        email: str = Field(description="email address")
        phone: Optional[str] = Field(description="phone number")
        address: Address = Field(description="contact address")
        company: Company = Field(description="company information")
    
    class ContactList(BaseModel):
        """Extract contact information from text"""
        contacts: List[Contact] = Field(description="List of contacts found")
    
    # Pattern 2: Conditional extraction
    class Event(BaseModel):
        """Extract event information"""
        event_type: str = Field(description="type of event (meeting, conference, etc.)")
        title: str = Field(description="event title")
        date: Optional[str] = Field(description="event date")
        location: Optional[str] = Field(description="event location")
        attendees: List[str] = Field(description="list of attendees")
    
    class EventExtraction(BaseModel):
        """Extract events from text"""
        events: List[Event] = Field(description="List of events found")
    
    print("Advanced extraction patterns available:")
    print("- Multi-level nested extraction")
    print("- Conditional field extraction")
    print("- Complex validation rules")
    print("- Batch processing with error handling")
    
    return ContactList, EventExtraction

# =============================================================================
# USAGE EXAMPLES & MAIN EXECUTION
# =============================================================================
# Run examples and demonstrate tagging and extraction functionality
# =============================================================================

if __name__ == "__main__":
    print("=== Tagging and Extraction Examples ===\n")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some examples may not work.")
        print("Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'\n")
    
    try:
        # Example 1: Basic tagging
        print("1. Basic Tagging Example:")
        basic_tagging_example()
        print()
        
        # Example 2: Basic extraction
        print("2. Basic Extraction Example:")
        basic_extraction_example()
        print()
        
        # Example 3: Document processing
        print("3. Document Processing Example:")
        document_processing_example()
        print()
        
        # Example 4: Paper extraction
        print("4. Paper Extraction Example:")
        paper_extraction_example()
        print()
        
        # Example 5: Batch processing
        print("5. Batch Processing Example:")
        batch_processing_example()
        print()
        
        # Example 6: Advanced patterns
        print("6. Advanced Patterns:")
        advanced_extraction_patterns()
        print()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")
    
    print("=== Tagging and Extraction Cheatsheet Complete ===")
    print("Key Concepts:")
    print("- Use Pydantic models to define extraction schemas")
    print("- Convert models to functions with convert_pydantic_to_openai_function")
    print("- Use JsonOutputFunctionsParser for clean output")
    print("- Use JsonKeyOutputFunctionsParser for specific key extraction")
    print("- Process large documents by splitting into chunks")
    print("- Use batch processing for efficient large-scale extraction")

