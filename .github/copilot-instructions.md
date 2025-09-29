# Copilot Instructions for Code Chat Bot

## Project Overview

This is a multi-AI chatbot application built with Streamlit that enables users to interact with multiple AI providers (OpenAI, MistralAI, Anthropic, and Cohere) through a unified interface. The application includes document processing capabilities using LangChain and provides cost tracking for API usage.

## Key Components

### Core Files
- `main.py` - Main Streamlit application with UI and AI provider integrations
- `document_loader.py` - Document processing and vector database functionality using LangChain
- `requirements.txt` - Python dependencies
- `.env` - Environment variables for API keys (not in repo)
- `previous_chats/` - Directory for storing chat history files
- `upload_docs/` - Directory for staging uploaded documents
- `total_cost.txt` - File tracking cumulative API costs

### Key Functions in main.py
- `calculate_cost(prompt, response, model)` - Calculate API usage costs
- `clear_session()` - Reset chat session state
- `save_state(filename)` - Save chat history to file
- `load_state(filename)` - Load previous chat history
- `add_cost_to_session_state(cost)` - Track costs in session
- `add_cost_to_total_cost(cost)` - Update cumulative cost file
- `perform_vector_db_search(file_extension, query, k)` - Search document embeddings

### DocumentLoader Class Methods
- `load_csv(file_path)` - Load CSV files
- `load_pdf(file_path)` - Load PDF documents
- `load_text(file_path)` - Load text files
- `load_web(url)` - Load web page content

### VectorDatabase Class Methods
- `split_document(document, chunk_size, chunk_overlap)` - Split documents for embedding
- `embed_upload(collection_name, split_documents)` - Create vector embeddings
- `search_vector_db(vector_store, query, k)` - Search similar documents

### Architecture
- **Frontend**: Streamlit web interface with sidebar controls
- **AI Providers**: OpenAI, MistralAI, Anthropic, Cohere APIs
- **Document Processing**: LangChain with Chroma vector database
- **File Support**: PDF, text, CSV files, and web URLs

## Development Guidelines

### Code Style
- Use clear, descriptive variable names
- Maintain consistent indentation (4 spaces)
- Add docstrings for functions and classes
- Follow Python PEP 8 guidelines
- Use type hints where appropriate

### API Integration Patterns
- Always use environment variables for API keys
- Implement proper error handling for API calls
- Include cost calculation for token usage
- Use streaming responses where supported
- Implement proper session state management for Streamlit

### Adding New AI Providers
1. Add API client initialization in the imports section
2. Update the `ai_providers` list in the UI section
3. Add model selection logic in the provider conditional blocks
4. Update the `cost` dictionary with pricing information
5. Implement the chat logic following existing patterns
6. Add proper error handling and cost tracking

### Document Processing
- Use LangChain document loaders for different file types
- Implement proper text splitting with appropriate chunk sizes
- Use OpenAI embeddings for vector storage
- Implement similarity search with relevance scoring

### Streamlit Best Practices
- Use session state for maintaining chat history
- Implement proper sidebar organization
- Use appropriate Streamlit components (chat_message, file_uploader, etc.)
- Handle file uploads in the `upload_docs` directory
- Implement proper error messages and user feedback

## Testing Guidelines

### Manual Testing
- Test each AI provider with basic queries
- Verify document upload and processing functionality
- Check cost calculations and session persistence
- Test model switching during conversations
- Validate error handling for missing API keys

### File Testing
- Test PDF, text, and CSV file uploads
- Verify web URL processing
- Check document search functionality
- Validate file cleanup and storage

## Environment Setup

### Required Environment Variables
```bash
OPENAI=your_openai_api_key
MISTRAL=your_mistral_api_key  
ANTHROPIC=your_anthropic_api_key
COHERE=your_cohere_api_key
```

### Dependencies
- Install requirements: `pip install -r requirements.txt`
- **Note**: `langchain-chroma` may need to be added to requirements.txt if not already included
- Run application: `streamlit run main.py`

## Common Issues and Solutions

### API Key Issues
- Verify all required API keys are set in `.env` file
- Check for typos in environment variable names
- Ensure `.env` file is in the root directory

### Document Processing Issues
- Check file upload limits and formats
- Verify OpenAI API key for embeddings
- Ensure proper file permissions in `upload_docs` directory

### Streamlit Issues
- Clear browser cache if UI doesn't update
- Check console for JavaScript errors
- Restart Streamlit server for configuration changes

### Model Availability
- Different providers may have different model availability
- Check provider documentation for model names and capabilities
- Update cost dictionary when new models are added

## Contributing Guidelines

### Before Making Changes
- Test the application locally
- Verify all AI providers work with sample queries
- Check that document processing functions correctly

### Code Changes
- Make minimal, focused changes
- Test with multiple AI providers
- Verify cost calculations remain accurate
- Update documentation if needed

### Adding Features
- Follow existing patterns for consistency
- Add appropriate error handling
- Update the UI to maintain good user experience
- Consider impact on session state and performance

## Security Considerations

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Validate file uploads for security
- Implement proper error handling to avoid exposing sensitive information
- Be mindful of cost implications when testing with real API keys

## Performance Considerations

- Implement proper session state management
- Use streaming responses for better user experience
- Consider caching for document embeddings
- Monitor token usage and costs
- Optimize chunk sizes for document processing

## Deployment Notes

- Ensure all environment variables are configured
- Set up proper file storage for document uploads
- Configure appropriate resource limits
- Monitor API usage and costs
- Implement proper logging for debugging