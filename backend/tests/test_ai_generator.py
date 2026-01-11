"""Tests for AIGenerator tool calling functionality"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator, MAX_TOOL_ROUNDS


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


class TestAIGeneratorSequentialToolCalling:
    """Test suite for sequential tool calling (multi-round)"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="test-model")
            return generator

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        manager = Mock()
        manager.execute_tool = Mock(return_value="Tool result content")
        return manager

    def _create_tool_use_response(self, tool_name="search_course_content", tool_id="tool_1", tool_input=None):
        """Helper to create a mock tool_use response"""
        response = Mock()
        response.stop_reason = "tool_use"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.id = tool_id
        tool_block.input = tool_input or {"query": "test"}

        response.content = [tool_block]
        return response

    def _create_text_response(self, text="Final answer"):
        """Helper to create a mock text response"""
        response = Mock()
        response.stop_reason = "end_turn"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = text

        response.content = [text_block]
        return response

    def test_single_tool_call_sufficient(self, ai_generator, mock_tool_manager):
        """Tool called once, Claude responds without further tools"""
        generator = ai_generator

        # API call 1: tool_use, API call 2: text response
        generator.client.messages.create.side_effect = [
            self._create_tool_use_response(),
            self._create_text_response("Answer after one tool call")
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify: 2 API calls, 1 tool execution
        assert generator.client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert result == "Answer after one tool call"

    def test_two_sequential_tool_calls(self, ai_generator, mock_tool_manager):
        """Claude uses tools twice before responding"""
        generator = ai_generator

        # API call 1: tool_use, API call 2: tool_use, API call 3: text
        generator.client.messages.create.side_effect = [
            self._create_tool_use_response(tool_id="tool_1", tool_input={"query": "MCP"}),
            self._create_tool_use_response(tool_id="tool_2", tool_input={"query": "Computer Use"}),
            self._create_text_response("Comparison of both courses")
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="Compare MCP and Computer Use courses",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify: 3 API calls, 2 tool executions
        assert generator.client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Comparison of both courses"

    def test_max_rounds_forces_final_response(self, ai_generator, mock_tool_manager):
        """After MAX_TOOL_ROUNDS, final call made without tools"""
        generator = ai_generator

        # Create exactly MAX_TOOL_ROUNDS tool_use responses for the loop
        # Plus 1 text response for the forced final call
        tool_responses = [
            self._create_tool_use_response(tool_id=f"tool_{i}")
            for i in range(MAX_TOOL_ROUNDS)
        ]
        # Last response after forced final call (without tools)
        tool_responses.append(self._create_text_response("Forced final response"))

        generator.client.messages.create.side_effect = tool_responses

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify: MAX_TOOL_ROUNDS tool executions
        assert mock_tool_manager.execute_tool.call_count == MAX_TOOL_ROUNDS

        # Verify: Final API call has no 'tools' parameter
        final_call = generator.client.messages.create.call_args_list[-1]
        assert "tools" not in final_call.kwargs

        assert result == "Forced final response"

    def test_tool_error_continues_gracefully(self, ai_generator, mock_tool_manager):
        """Tool exception captured, processing continues"""
        generator = ai_generator

        # Make tool execution raise an exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        generator.client.messages.create.side_effect = [
            self._create_tool_use_response(),
            self._create_text_response("Response after tool error")
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was attempted
        assert mock_tool_manager.execute_tool.call_count == 1

        # Verify error was captured in tool_result (check the second API call)
        second_call = generator.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_result = messages[2]["content"][0]
        assert tool_result["is_error"] is True
        assert "Error" in tool_result["content"]

        # Verify response still generated
        assert result == "Response after tool error"

    def test_no_tool_use_returns_immediately(self, ai_generator, mock_tool_manager):
        """When Claude doesn't use tools, return directly"""
        generator = ai_generator

        # API returns text without tool use
        generator.client.messages.create.return_value = self._create_text_response("Direct answer")

        tools = [{"name": "search_course_content", "description": "Search"}]

        result = generator.generate_response(
            query="What is 2+2?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify: 1 API call, 0 tool executions
        assert generator.client.messages.create.call_count == 1
        assert mock_tool_manager.execute_tool.call_count == 0
        assert result == "Direct answer"

    def test_messages_accumulate_across_rounds(self, ai_generator, mock_tool_manager):
        """Messages accumulate correctly across tool rounds"""
        generator = ai_generator

        generator.client.messages.create.side_effect = [
            self._create_tool_use_response(tool_id="tool_1"),
            self._create_tool_use_response(tool_id="tool_2"),
            self._create_text_response("Final")
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]

        generator.generate_response(
            query="test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Check third API call has accumulated messages
        third_call = generator.client.messages.create.call_args_list[2]
        messages = third_call.kwargs["messages"]

        # Should have: user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"


class TestAIGeneratorHelperMethods:
    """Test helper methods"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(api_key="test_key", model="test-model")
            return generator

    def test_extract_text_from_response(self, ai_generator):
        """Test _extract_text extracts text correctly"""
        generator = ai_generator

        response = Mock()
        text_block = Mock()
        text_block.text = "Extracted text"
        response.content = [text_block]

        result = generator._extract_text(response)
        assert result == "Extracted text"

    def test_extract_text_empty_content(self, ai_generator):
        """Test _extract_text handles empty content"""
        generator = ai_generator

        response = Mock()
        response.content = []

        result = generator._extract_text(response)
        assert result == "Unable to generate a response."

    def test_extract_text_no_text_block(self, ai_generator):
        """Test _extract_text handles no text blocks"""
        generator = ai_generator

        response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        # No 'text' attribute
        del tool_block.text
        response.content = [tool_block]

        result = generator._extract_text(response)
        assert result == "Unable to generate a response."

    def test_execute_tools_success(self, ai_generator):
        """Test _execute_tools executes successfully"""
        generator = ai_generator

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool output"

        response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        response.content = [tool_block]

        results, has_error = generator._execute_tools(response, tool_manager)

        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tool_123"
        assert results[0]["content"] == "Tool output"
        assert has_error is False

    def test_execute_tools_with_error(self, ai_generator):
        """Test _execute_tools handles exceptions"""
        generator = ai_generator

        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = Exception("Tool crashed")

        response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        response.content = [tool_block]

        results, has_error = generator._execute_tools(response, tool_manager)

        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert "Error" in results[0]["content"]
        assert has_error is True

    def test_execute_tools_multiple(self, ai_generator):
        """Test _execute_tools handles multiple tool calls"""
        generator = ai_generator

        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Output"

        response = Mock()
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "query1"}

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"course_name": "MCP"}

        response.content = [tool_block_1, tool_block_2]

        results, has_error = generator._execute_tools(response, tool_manager)

        assert len(results) == 2
        assert tool_manager.execute_tool.call_count == 2
        assert has_error is False
