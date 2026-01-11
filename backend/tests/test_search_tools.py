"""Tests for CourseSearchTool execute method"""
import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def test_execute_with_valid_query(self, mock_vector_store, mock_search_results):
        """Test basic search returns formatted results"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="What is MCP?")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains content
        assert "MCP stands for Model Context Protocol" in result
        assert "[MCP: Build Rich-Context AI Apps" in result

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test search with course name filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="architecture", course_name="MCP")

        mock_vector_store.search.assert_called_once_with(
            query="architecture",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test search with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="introduction", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test search with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="content", course_name="MCP", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_empty_results(self, mock_vector_store, mock_empty_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_course_filter(self, mock_vector_store, mock_empty_results):
        """Test empty results message includes course filter info"""
        mock_vector_store.search.return_value = mock_empty_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent", course_name="Test Course")

        assert "No relevant content found" in result
        assert "Test Course" in result

    def test_execute_error_handling(self, mock_vector_store, mock_error_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = mock_error_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test query")

        assert "Search error" in result or "Connection failed" in result

    def test_format_results_includes_headers(self, mock_vector_store, mock_search_results):
        """Test that results include course/lesson headers"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="MCP")

        # Should have headers with course title and lesson numbers
        assert "[MCP: Build Rich-Context AI Apps - Lesson 1]" in result
        assert "[MCP: Build Rich-Context AI Apps - Lesson 2]" in result

    def test_sources_tracked(self, mock_vector_store, mock_search_results):
        """Test that sources are tracked after execution"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="MCP")

        # Check sources were stored
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "MCP: Build Rich-Context AI Apps - Lesson 1"

    def test_sources_include_links(self, mock_vector_store, mock_search_results):
        """Test that sources include links when available"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="MCP")

        # Verify links are included
        assert tool.last_sources[0]["link"] is not None


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    def test_execute_returns_outline(self, mock_vector_store):
        """Test outline tool returns formatted course structure"""
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="MCP")

        assert "Course: MCP: Build Rich-Context AI Apps" in result
        assert "Link:" in result
        assert "Lessons:" in result
        assert "Introduction" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course not found"""
        mock_vector_store.get_course_outline.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Nonexistent Course")

        assert "No course found" in result


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store, mock_search_results):
        """Test executing a registered tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert result is not None
        mock_vector_store.search.assert_called_once()

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result
