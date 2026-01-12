import anthropic
from typing import List, Optional, Dict, Any, Tuple

# Maximum sequential tool call rounds per query
MAX_TOOL_ROUNDS = 2


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course content for specific topics or detailed information
2. **get_course_outline**: Get course structure - title, course link, and complete lesson list with numbers and titles

Tool Selection Rules (IMPORTANT - follow these strictly):
- Use **get_course_outline** when the user asks about:
  - "what lessons" or "how many lessons" in a course
  - "outline" or "structure" of a course
  - "syllabus" or "table of contents"
  - "what does the course cover" or "course overview"
  - listing all topics/lessons in a course
- Use **search_course_content** when the user asks about:
  - specific topics or concepts (e.g., "what is MCP architecture")
  - detailed explanations from lesson content
  - specific information within lessons

Tool Usage:
- You may use tools sequentially if needed (up to 2 rounds)
- After each tool result, evaluate if you have enough information to answer
- If first search is insufficient, refine your search or use a different tool
- Synthesize results from all tool calls into a coherent response

When to use multiple tool calls:
- Question spans multiple courses or lessons
- Need both course outline AND content search
- First search returned partial results that need refinement

Response Protocol for Outline Queries:
- Always include the course title and course link
- List all lessons with their numbers and titles
- Present the information in a clear, organized format

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**: Provide direct answers only — no reasoning process or tool explanations

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content with conversation history
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message list
        messages = [{"role": "user", "content": query}]

        # Track tool call rounds
        round_count = 0

        # Main tool execution loop
        while round_count < MAX_TOOL_ROUNDS:
            # Build API params - include tools to allow tool use
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self.client.messages.create(**api_params)

            # Termination: No tool use requested - return text response
            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            # Termination: No tool manager available
            if not tool_manager:
                return self._extract_text(response)

            # Execute tools and collect results
            tool_results, has_error = self._execute_tools(response, tool_manager)

            # Append assistant's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Append tool results as user message
            messages.append({"role": "user", "content": tool_results})

            # Increment round counter
            round_count += 1

        # Max rounds reached - make final call WITHOUT tools to force text response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _execute_tools(self, response, tool_manager) -> Tuple[List[Dict], bool]:
        """
        Execute all tool calls in a response.

        Args:
            response: The API response containing tool use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results list, has_error boolean)
        """
        tool_results = []
        has_error = False

        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    has_error = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        }
                    )

        return tool_results, has_error

    def _extract_text(self, response) -> str:
        """
        Extract text from response content blocks.

        Args:
            response: The API response

        Returns:
            Text content or fallback message
        """
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        return "Unable to generate a response."
