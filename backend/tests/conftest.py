"""Shared fixtures for RAG chatbot tests"""
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


@pytest.fixture
def mock_search_results():
    """Create mock search results with sample data"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It enables AI applications to connect to tools.",
            "The MCP architecture consists of clients and servers that communicate via JSON-RPC."
        ],
        metadata=[
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 2, "chunk_index": 5}
        ],
        distances=[0.25, 0.35]
    )


@pytest.fixture
def mock_empty_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_error_results():
    """Create search results with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: Connection failed"
    )


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Create a mock VectorStore"""
    store = Mock()
    store.search = Mock(return_value=mock_search_results)
    store.get_course_link = Mock(return_value="https://example.com/course")
    store.get_lesson_link = Mock(return_value="https://example.com/course/lesson1")
    store.get_course_outline = Mock(return_value={
        "title": "MCP: Build Rich-Context AI Apps",
        "course_link": "https://example.com/course",
        "instructor": "Test Instructor",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction"},
            {"lesson_number": 1, "lesson_title": "Why MCP"},
        ]
    })
    return store


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    manager = Mock()
    manager.execute_tool = Mock(return_value="[MCP Course - Lesson 1]\nMCP content here...")
    manager.get_tool_definitions = Mock(return_value=[
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ])
    manager.get_last_sources = Mock(return_value=[])
    manager.reset_sources = Mock()
    return manager


@pytest.fixture
def mock_anthropic_response_with_tool_use():
    """Create mock Anthropic response that requests tool use"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "MCP architecture"}

    response.content = [tool_use_block]
    return response


@pytest.fixture
def mock_anthropic_response_text():
    """Create mock Anthropic response with text"""
    response = Mock()
    response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = "MCP is a protocol for connecting AI to tools."

    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_text):
    """Create a mock Anthropic client"""
    client = Mock()
    client.messages = Mock()
    client.messages.create = Mock(return_value=mock_anthropic_response_text)
    return client
