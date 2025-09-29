# Code Chat Bot - Modernized Architecture

This project has been modernized with:

## New Project Structure

```
src/code_chat_bot/
├── __init__.py                 # Main module exports
├── config/
│   └── __init__.py            # Configuration management with Pydantic
├── models/
│   └── __init__.py            # Pydantic data models
├── providers/
│   └── __init__.py            # AI provider abstractions
└── document_processing/
    └── __init__.py            # Document processing with validation
```

## Key Improvements

1. **Modern Package Management**: Uses `uv` for dependency management
2. **Pydantic Integration**: All data structures use Pydantic for validation
3. **Modular Architecture**: Code is organized into logical modules
4. **Type Safety**: Full typing support with Pydantic models
5. **Configuration Management**: Centralized config with validation

## Usage with uv

Install dependencies:
```bash
uv sync
```

Run the application:
```bash
uv run streamlit run main.py
```

## Migration Notes

- Original files are preserved as `main_original.py` and `document_loader_original.py`
- All functionality is maintained but now uses modern patterns
- Configuration is now centralized and validated
- API responses are validated with Pydantic models