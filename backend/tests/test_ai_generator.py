"""Tests for AIGenerator tool calling functionality"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="test-model")
            return generator, mock_anthropic

    def test_tools_passed_to_api(self, ai_generator, mock_anthropic_response_text):
        """Test that tools are passed to Claude API call"""
        generator, mock_anthropic = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        tools = [{"name": "search_course_content", "description": "Search"}]

        generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=None
        )

        # Verify API was called with tools
        call_args = generator.client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools

    def test_tool_choice_set_to_auto(self, ai_generator, mock_anthropic_response_text):
        """Test that tool_choice is set to auto when tools provided"""
        generator, _ = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        tools = [{"name": "search_course_content", "description": "Search"}]

        generator.generate_response(
            query="test",
            tools=tools,
            tool_manager=None
        )

        call_args = generator.client.messages.create.call_args
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    def test_tool_use_detected(self, ai_generator, mock_anthropic_response_with_tool_use, mock_tool_manager):
        """Test that tool use is detected from response"""
        generator, _ = ai_generator

        # First call returns tool_use, second call returns text
        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.text = "Here is the answer about MCP."
        text_response.content = [text_block]

        generator.client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            text_response
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool_manager.execute_tool was called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP architecture"
        )

    def test_tool_execution_result_sent_back(self, ai_generator, mock_anthropic_response_with_tool_use, mock_tool_manager):
        """Test that tool execution results are sent back to Claude"""
        generator, _ = ai_generator

        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.text = "Final answer"
        text_response.content = [text_block]

        generator.client.messages.create.side_effect = [
            mock_anthropic_response_with_tool_use,
            text_response
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Second API call should include tool results
        second_call = generator.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"

    def test_final_response_extracted(self, ai_generator, mock_anthropic_response_text):
        """Test that final text response is correctly extracted"""
        generator, _ = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        result = generator.generate_response(query="Hello")

        assert result == "MCP is a protocol for connecting AI to tools."

    def test_no_tools_provided(self, ai_generator, mock_anthropic_response_text):
        """Test response generation without tools"""
        generator, _ = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        result = generator.generate_response(query="Hello")

        call_args = generator.client.messages.create.call_args
        assert "tools" not in call_args.kwargs

    def test_conversation_history_included(self, ai_generator, mock_anthropic_response_text):
        """Test that conversation history is included in system prompt"""
        generator, _ = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        generator.generate_response(
            query="Follow up question",
            conversation_history="User: Hi\nAssistant: Hello!"
        )

        call_args = generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert "User: Hi" in system_content

    def test_system_prompt_included(self, ai_generator, mock_anthropic_response_text):
        """Test that system prompt is always included"""
        generator, _ = ai_generator
        generator.client.messages.create.return_value = mock_anthropic_response_text

        generator.generate_response(query="test")

        call_args = generator.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "AI assistant specialized in course materials" in system_content


class TestAIGeneratorToolExecution:
    """Test the _handle_tool_execution method"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="test-model")
            return generator

    def test_multiple_tool_calls_handled(self, ai_generator, mock_tool_manager):
        """Test handling of multiple tool calls in single response"""
        generator = ai_generator

        # Create response with two tool calls
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_1"
        tool_use_1.input = {"query": "query1"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "search_course_content"
        tool_use_2.id = "tool_2"
        tool_use_2.input = {"query": "query2"}

        initial_response = Mock()
        initial_response.content = [tool_use_1, tool_use_2]

        text_response = Mock()
        text_block = Mock()
        text_block.text = "Combined answer"
        text_response.content = [text_block]

        generator.client.messages.create.return_value = text_response

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }

        result = generator._handle_tool_execution(
            initial_response,
            base_params,
            mock_tool_manager
        )

        # Both tools should be executed
        assert mock_tool_manager.execute_tool.call_count == 2
