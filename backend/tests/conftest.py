"""Shared fixtures for RAG chatbot tests"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API tests"""
    rag = Mock()

    # Mock query method
    rag.query = Mock(return_value=(
        "MCP is a protocol for connecting AI applications to tools.",
        [{"text": "MCP Course - Lesson 1", "link": "https://example.com/mcp"}]
    ))

    # Mock session manager
    rag.session_manager = Mock()
    rag.session_manager.create_session = Mock(return_value="session_123")

    # Mock get_course_analytics
    rag.get_course_analytics = Mock(return_value={
        "total_courses": 3,
        "course_titles": ["MCP Course", "Computer Use", "Claude API"]
    })

    # Mock add_course_folder
    rag.add_course_folder = Mock(return_value=(2, 50))

    return rag


@pytest.fixture
def mock_config():
    """Create mock configuration for tests"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test_key"
    config.ANTHROPIC_MODEL = "test-model"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System - Test")

    # Pydantic models (same as app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Store rag_system reference for the routes
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag_system = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag_system = app.state.rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for API testing"""
    from starlette.testclient import TestClient
    return TestClient(test_app)


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is MCP?",
        "session_id": None
    }


@pytest.fixture
def sample_query_with_session():
    """Sample query request with session ID"""
    return {
        "query": "Tell me more about lesson 2",
        "session_id": "session_existing_123"
    }


@pytest.fixture
def mock_search_results():
    """Create mock search results with sample data"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It enables AI applications to connect to tools.",
            "The MCP architecture consists of clients and servers that communicate via JSON-RPC.",
        ],
        metadata=[
            {
                "course_title": "MCP: Build Rich-Context AI Apps",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "MCP: Build Rich-Context AI Apps",
                "lesson_number": 2,
                "chunk_index": 5,
            },
        ],
        distances=[0.25, 0.35],
    )


@pytest.fixture
def mock_empty_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_error_results():
    """Create search results with error"""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Search error: Connection failed"
    )


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Create a mock VectorStore"""
    store = Mock()
    store.search = Mock(return_value=mock_search_results)
    store.get_course_link = Mock(return_value="https://example.com/course")
    store.get_lesson_link = Mock(return_value="https://example.com/course/lesson1")
    store.get_course_outline = Mock(
        return_value={
            "title": "MCP: Build Rich-Context AI Apps",
            "course_link": "https://example.com/course",
            "instructor": "Test Instructor",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction"},
                {"lesson_number": 1, "lesson_title": "Why MCP"},
            ],
        }
    )
    return store


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    manager = Mock()
    manager.execute_tool = Mock(
        return_value="[MCP Course - Lesson 1]\nMCP content here..."
    )
    manager.get_tool_definitions = Mock(
        return_value=[
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )
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
