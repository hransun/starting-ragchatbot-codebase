"""Tests for RAGSystem end-to-end query handling"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGSystemQuery:
    """Test suite for RAGSystem.query() method"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
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
    def rag_system(self, mock_config):
        """Create RAGSystem with mocked dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_session:

            # Setup mock vector store
            mock_vs_instance = Mock()
            mock_vs_instance.search = Mock()
            mock_vs_instance.get_course_link = Mock(return_value="https://example.com")
            mock_vs_instance.get_lesson_link = Mock(return_value="https://example.com/lesson1")
            mock_vs_instance.get_course_outline = Mock(return_value=None)
            mock_vs.return_value = mock_vs_instance

            # Setup mock AI generator
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response = Mock(return_value="Test response about MCP")
            mock_ai.return_value = mock_ai_instance

            # Setup mock session manager
            mock_session_instance = Mock()
            mock_session_instance.get_conversation_history = Mock(return_value=None)
            mock_session_instance.add_exchange = Mock()
            mock_session.return_value = mock_session_instance

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            return system, mock_ai_instance, mock_session_instance

    def test_query_calls_ai_generator(self, rag_system):
        """Test that query calls AI generator with correct parameters"""
        system, mock_ai, _ = rag_system

        system.query("What is MCP?")

        mock_ai.generate_response.assert_called_once()
        call_args = mock_ai.generate_response.call_args
        assert "MCP" in call_args.kwargs["query"]

    def test_query_passes_tools(self, rag_system):
        """Test that query passes tool definitions to AI"""
        system, mock_ai, _ = rag_system

        system.query("What is MCP?")

        call_args = mock_ai.generate_response.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] is not None

    def test_query_passes_tool_manager(self, rag_system):
        """Test that query passes tool manager to AI"""
        system, mock_ai, _ = rag_system

        system.query("What is MCP?")

        call_args = mock_ai.generate_response.call_args
        assert "tool_manager" in call_args.kwargs
        assert call_args.kwargs["tool_manager"] is not None

    def test_query_returns_response_and_sources(self, rag_system):
        """Test that query returns tuple of response and sources"""
        system, _, _ = rag_system

        response, sources = system.query("What is MCP?")

        assert response == "Test response about MCP"
        assert isinstance(sources, list)

    def test_query_with_session_gets_history(self, rag_system):
        """Test that query with session ID retrieves history"""
        system, mock_ai, mock_session = rag_system
        mock_session.get_conversation_history.return_value = "Previous chat"

        system.query("Follow up", session_id="session_1")

        mock_session.get_conversation_history.assert_called_with("session_1")
        call_args = mock_ai.generate_response.call_args
        assert call_args.kwargs["conversation_history"] == "Previous chat"

    def test_query_updates_session_history(self, rag_system):
        """Test that query updates session history after response"""
        system, _, mock_session = rag_system

        system.query("What is MCP?", session_id="session_1")

        mock_session.add_exchange.assert_called_once_with(
            "session_1",
            "What is MCP?",
            "Test response about MCP"
        )

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """Test that sources are reset after being retrieved"""
        system, _, _ = rag_system

        # Replace reset_sources with a mock to track calls
        from unittest.mock import Mock
        system.tool_manager.reset_sources = Mock()

        system.query("What is MCP?")

        system.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemToolIntegration:
    """Test tool integration in RAGSystem"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
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

    def test_search_tool_registered(self, mock_config):
        """Test that CourseSearchTool is registered"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            assert "search_course_content" in system.tool_manager.tools

    def test_outline_tool_registered(self, mock_config):
        """Test that CourseOutlineTool is registered"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            assert "get_course_outline" in system.tool_manager.tools

    def test_tool_definitions_available(self, mock_config):
        """Test that tool definitions are available for API"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            definitions = system.tool_manager.get_tool_definitions()

            assert len(definitions) == 2
            tool_names = [d["name"] for d in definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names


class TestRAGSystemErrorHandling:
    """Test error handling in RAGSystem"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
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

    def test_query_handles_ai_exception(self, mock_config):
        """Test that query handles AI generator exceptions"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager'):

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response = Mock(side_effect=Exception("API Error"))
            mock_ai.return_value = mock_ai_instance

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            # Should raise or handle gracefully
            with pytest.raises(Exception):
                system.query("test")
