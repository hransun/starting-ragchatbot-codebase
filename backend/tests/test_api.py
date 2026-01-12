"""Tests for FastAPI endpoints"""
import pytest
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQueryEndpoint:
    """Test suite for POST /api/query endpoint"""

    def test_query_success_without_session(self, test_client, test_app, sample_query_request):
        """Test successful query without existing session"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "session_123"

    def test_query_success_with_session(self, test_client, test_app, sample_query_with_session):
        """Test successful query with existing session ID"""
        response = test_client.post("/api/query", json=sample_query_with_session)

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_existing_123"

    def test_query_returns_answer(self, test_client, test_app, sample_query_request):
        """Test that query returns expected answer from RAG system"""
        response = test_client.post("/api/query", json=sample_query_request)

        data = response.json()
        assert "MCP is a protocol" in data["answer"]

    def test_query_returns_sources(self, test_client, test_app, sample_query_request):
        """Test that query returns sources list"""
        response = test_client.post("/api/query", json=sample_query_request)

        data = response.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "MCP Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/mcp"

    def test_query_calls_rag_system(self, test_client, test_app, sample_query_request):
        """Test that endpoint calls RAG system query method"""
        test_client.post("/api/query", json=sample_query_request)

        rag_system = test_app.state.rag_system
        rag_system.query.assert_called_once()

    def test_query_passes_query_text(self, test_client, test_app, sample_query_request):
        """Test that query text is passed to RAG system"""
        test_client.post("/api/query", json=sample_query_request)

        rag_system = test_app.state.rag_system
        call_args = rag_system.query.call_args
        assert call_args[0][0] == "What is MCP?"

    def test_query_missing_query_field(self, test_client):
        """Test that missing query field returns 422 error"""
        response = test_client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_empty_query(self, test_client):
        """Test query with empty string"""
        response = test_client.post("/api/query", json={"query": ""})

        # Empty query is valid per schema, endpoint should handle it
        assert response.status_code == 200

    def test_query_creates_session_when_none_provided(self, test_client, test_app, sample_query_request):
        """Test that new session is created when none provided"""
        test_client.post("/api/query", json=sample_query_request)

        rag_system = test_app.state.rag_system
        rag_system.session_manager.create_session.assert_called_once()

    def test_query_preserves_existing_session(self, test_client, test_app, sample_query_with_session):
        """Test that existing session ID is preserved"""
        test_client.post("/api/query", json=sample_query_with_session)

        rag_system = test_app.state.rag_system
        # Should not create new session when one is provided
        rag_system.session_manager.create_session.assert_not_called()

    def test_query_handles_rag_exception(self, test_client, test_app, sample_query_request):
        """Test that RAG system exceptions return 500 error"""
        rag_system = test_app.state.rag_system
        rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_with_special_characters(self, test_client, test_app):
        """Test query with special characters"""
        request = {"query": "What's the <script>alert('test')</script> lesson?"}
        response = test_client.post("/api/query", json=request)

        assert response.status_code == 200

    def test_query_with_unicode(self, test_client, test_app):
        """Test query with unicode characters"""
        request = {"query": "What about 日本語 content?"}
        response = test_client.post("/api/query", json=request)

        assert response.status_code == 200


class TestCoursesEndpoint:
    """Test suite for GET /api/courses endpoint"""

    def test_courses_success(self, test_client, test_app):
        """Test successful courses endpoint response"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_courses_returns_correct_count(self, test_client, test_app):
        """Test that courses returns correct count"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert data["total_courses"] == 3

    def test_courses_returns_titles_list(self, test_client, test_app):
        """Test that courses returns list of titles"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert len(data["course_titles"]) == 3
        assert "MCP Course" in data["course_titles"]
        assert "Computer Use" in data["course_titles"]
        assert "Claude API" in data["course_titles"]

    def test_courses_calls_analytics(self, test_client, test_app):
        """Test that endpoint calls get_course_analytics"""
        test_client.get("/api/courses")

        rag_system = test_app.state.rag_system
        rag_system.get_course_analytics.assert_called_once()

    def test_courses_handles_exception(self, test_client, test_app):
        """Test that analytics exceptions return 500 error"""
        rag_system = test_app.state.rag_system
        rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]

    def test_courses_empty_catalog(self, test_client, test_app):
        """Test response when no courses exist"""
        rag_system = test_app.state.rag_system
        rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestRootEndpoint:
    """Test suite for GET / endpoint"""

    def test_root_returns_ok(self, test_client):
        """Test that root endpoint returns ok status"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_returns_message(self, test_client):
        """Test that root endpoint returns message"""
        response = test_client.get("/")

        data = response.json()
        assert "message" in data


class TestAPIResponseFormat:
    """Test API response format and headers"""

    def test_query_response_content_type(self, test_client, sample_query_request):
        """Test that query response has correct content type"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.headers["content-type"] == "application/json"

    def test_courses_response_content_type(self, test_client):
        """Test that courses response has correct content type"""
        response = test_client.get("/api/courses")

        assert response.headers["content-type"] == "application/json"

    def test_invalid_endpoint_returns_404(self, test_client):
        """Test that invalid endpoints return 404"""
        response = test_client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_wrong_method_returns_405(self, test_client):
        """Test that wrong HTTP method returns 405"""
        response = test_client.get("/api/query")

        assert response.status_code == 405


class TestQueryRequestValidation:
    """Test request validation for query endpoint"""

    def test_invalid_json_returns_error(self, test_client):
        """Test that invalid JSON returns error"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_type_for_query(self, test_client):
        """Test that wrong type for query field returns error"""
        response = test_client.post("/api/query", json={"query": 123})

        assert response.status_code == 422

    def test_wrong_type_for_session_id(self, test_client, test_app):
        """Test that wrong type for session_id field returns error"""
        response = test_client.post("/api/query", json={
            "query": "test",
            "session_id": 123
        })

        assert response.status_code == 422

    def test_extra_fields_ignored(self, test_client, test_app):
        """Test that extra fields are ignored"""
        response = test_client.post("/api/query", json={
            "query": "test",
            "extra_field": "ignored"
        })

        assert response.status_code == 200
