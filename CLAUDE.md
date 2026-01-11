# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot system for querying course materials. Built with FastAPI backend, vanilla JavaScript frontend, ChromaDB vector store, and Anthropic's Claude AI with tool calling.

## Development Commands

### Installation
```bash
uv sync  # Install all Python dependencies
```

### Running the Application
```bash
./run.sh  # Quick start (creates docs folder, starts server)
# OR
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Running on Windows (Git Bash)
```bash
cd /c/repo/starting-ragchatbot-codebase/backend && /c/repo/starting-ragchatbot-codebase/.venv/Scripts/python.exe -m uvicorn app:app --reload --port 8000
```

### Environment Setup
Create `.env` in root directory:
```
ANTHROPIC_API_KEY=your_api_key_here
```

### Access Points
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture Overview

### Request Flow (User Query → Response)

1. **Frontend** (`frontend/script.js:45-96`)
   - User input → `sendMessage()` → POST `/api/query` with `{query, session_id}`

2. **FastAPI Endpoint** (`backend/app.py:56-74`)
   - Receives query → Manages session → Calls `rag_system.query()`

3. **RAG System** (`backend/rag_system.py:102-140`)
   - Orchestrates: session history → AI generator → extract sources → update session

4. **AI Generator** (`backend/ai_generator.py`)
   - Sends query + conversation history + tools to Claude API
   - Handles tool execution loop (search requests)
   - Returns final response after processing tool results

5. **Search Tool** (`backend/search_tools.py:52-86`)
   - Executes when AI decides to search course content
   - Calls vector store with optional course/lesson filters
   - Formats results with context headers `[Course Title - Lesson X]`

6. **Vector Store** (`backend/vector_store.py:61-100`)
   - Two ChromaDB collections:
     - `course_catalog`: Course metadata for fuzzy name matching
     - `course_content`: Text chunks with embeddings
   - Performs semantic search using `all-MiniLM-L6-v2` embeddings
   - Returns top 5 results (configurable)

### Document Processing Pipeline

**Entry Point**: `backend/app.py:88-98` (startup loads from `../docs` folder)

**Flow**: `rag_system.py:52-100` → `document_processor.py:97-259` → `vector_store.py:162-180`

1. **Parse** (`document_processor.py:97-259`)
   - Expected format:
     ```
     Course Title: [title]
     Course Link: [url]
     Course Instructor: [name]

     Lesson 0: [title]
     Lesson Link: [optional]
     [content...]
     ```
   - Extracts course metadata + lesson structure using regex

2. **Chunk** (`document_processor.py:25-91`)
   - Sentence-based splitting (handles abbreviations)
   - 800 char chunks, 100 char overlap
   - Adds context: `"Course {title} Lesson {number} content: {chunk}"`

3. **Embed & Store** (`vector_store.py:162-180`)
   - Generates embeddings via SentenceTransformers
   - Stores in ChromaDB with metadata (course_title, lesson_number, chunk_index)

### Key Components

**Session Management** (`backend/session_manager.py`)
- In-memory dictionary storage
- Maintains last 5 conversation exchanges (configurable via `config.MAX_HISTORY`)
- Session format: `"session_1"`, `"session_2"`, etc.

**Configuration** (`backend/config.py`)
- Centralized settings: API keys, model names, chunk sizes, search limits
- Important constants:
  - `CHUNK_SIZE`: 800
  - `CHUNK_OVERLAP`: 100
  - `MAX_RESULTS`: 5 (search results)
  - `MAX_HISTORY`: 2 (conversation pairs)
  - `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"

**Data Models** (`backend/models.py`)
- `Course`: title, link, instructor, lessons[]
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content, course_title, lesson_number, chunk_index

## Important Implementation Details

### Tool Calling Pattern
The AI uses function calling to search. Flow in `ai_generator.py:89-135`:
1. Claude returns `tool_use` block
2. Tool executed via `tool_manager.execute_tool()`
3. Results added to conversation
4. Second Claude API call made with tool results
5. Final answer generated

### Frontend-Backend Communication
- **Protocol**: REST API with JSON
- **No WebSockets**: Simple HTTP POST/Response
- **CORS**: Enabled for cross-origin requests (`app.py:24-32`)
- **Static Files**: Frontend served from `/` by FastAPI (`app.py:79-82`)

### Vector Search Features
- **Fuzzy Course Matching**: Partial course names work (e.g., "MCP" matches full title)
- **Filtering**: Supports course name AND/OR lesson number filters
- **Duplicate Prevention**: Checks existing courses before re-ingestion (`rag_system.py:75-96`)

### ChromaDB Collections
Both collections use persistent storage at `./chroma_db/`:
- `course_catalog`: One document per course (ID = course title)
- `course_content`: Multiple documents per course (ID = `{course_title}_{chunk_index}`)

## Code Modification Guidelines

### Adding New Course Document Sources
1. Update `document_processor.py:13-21` to handle new file formats
2. Supported: `.txt`, `.pdf`, `.docx` (currently only `.txt` fully implemented)

### Changing Chunking Strategy
- Modify `document_processor.py:25-91` (`chunk_text()` method)
- Update `config.py` constants: `CHUNK_SIZE`, `CHUNK_OVERLAP`

### Adding New Search Filters
1. Add parameters to `CourseSearchTool` definition in `search_tools.py:27-50`
2. Update `vector_store.py:118-133` (`_build_filter()`) to handle new filter types

### Modifying AI Behavior
- System prompt: `ai_generator.py:7-30`
- Model selection: `config.py:13`
- Temperature/tokens: `ai_generator.py:71-73`

### Session History Management
- Conversation format: `session_manager.py:42-56` (`get_conversation_history()`)
- History limit: `config.py:22` (`MAX_HISTORY`)

## File Organization

```
backend/
├── app.py                 # FastAPI server, endpoints, startup
├── rag_system.py          # Main orchestrator
├── ai_generator.py        # Claude API client + tool execution
├── search_tools.py        # Course search tool definition
├── vector_store.py        # ChromaDB interface
├── document_processor.py  # Parsing + chunking
├── session_manager.py     # Conversation history
├── models.py              # Data models
└── config.py              # Centralized configuration

frontend/
├── index.html             # UI structure
├── script.js              # Query handling, message rendering
└── styles.css             # Styling

docs/                      # Course documents loaded on startup
```

## Testing Queries

The system works best with course-specific questions. Example queries:
- "What is covered in lesson 2 of the MCP course?"
- "Tell me about computer use with Anthropic"
- "What does Colt Steele teach?"

The AI will automatically use the search tool when needed and can filter by course name and/or lesson number based on the query context.
