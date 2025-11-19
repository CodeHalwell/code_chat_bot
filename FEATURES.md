# Enhanced Features Documentation

This document describes all the enhanced features added to the AI ChatBot application.

## Table of Contents
1. [Latest AI Models](#latest-ai-models)
2. [Database Integration](#database-integration)
3. [Voice Capabilities](#voice-capabilities)
4. [NLP Analysis](#nlp-analysis)
5. [Multi-language Support](#multi-language-support)
6. [Logging & Monitoring](#logging--monitoring)
7. [AI Agents](#ai-agents)
8. [Enhanced RAG](#enhanced-rag)

## Latest AI Models

### OpenAI
- **GPT-4o**: Most advanced multimodal model with vision
- **GPT-4o-mini**: Small, affordable model for fast tasks
- **o1-preview**: Advanced reasoning model
- **o1-mini**: Fast reasoning for coding and STEM
- **GPT-4 Turbo**: Improved instruction following
- **GPT-4**: Original most capable model
- **GPT-3.5 Turbo**: Fast and economical

### Anthropic (Claude)
- **Claude 3.5 Sonnet**: Latest model with best-in-class coding
- **Claude 3 Opus**: Most powerful for complex tasks
- **Claude 3 Sonnet**: Balanced performance
- **Claude 3 Haiku**: Fastest and most compact

### MistralAI
- **Mistral Large**: Flagship model
- **Mistral Medium**: Balanced performance
- **Mistral Small**: Cost-effective
- **Mixtral 8x22B & 8x7B**: Sparse MoE models
- **Mistral 7B**: Base fast model

### Cohere
- **Command R+**: Most powerful for RAG
- **Command R**: Optimized for long-context
- **Command**: General purpose
- **Command Light**: Fast and lightweight

### Google (NEW!)
- **Gemini 1.5 Pro**: 2M token context window
- **Gemini 1.5 Flash**: Fast and efficient
- **Gemini Pro**: Versatile model

## Database Integration

### Supported Databases
1. **PostgreSQL**: Enterprise-grade relational database
2. **MongoDB**: NoSQL document database
3. **Firebase**: Cloud-hosted database
4. **JSON**: File-based fallback (default)

### Features
- Automatic chat session persistence
- Session metadata tracking
- Query and restore previous conversations
- List recent chat sessions

### Configuration
Set via environment variables:
```bash
# PostgreSQL
export DB_TYPE=postgresql
export POSTGRES_HOST=localhost
export POSTGRES_DB=chatbot
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=password

# MongoDB
export DB_TYPE=mongodb
export MONGODB_URI=mongodb://localhost:27017/

# JSON (default)
export DB_TYPE=json
```

## Voice Capabilities

### Voice Input
- **Speech Recognition**: Convert speech to text
- **Multiple Recognition Engines**: Google Speech Recognition (free)
- **Language Support**: Configurable language detection
- **Live Microphone Input**: Real-time voice commands

### Voice Output
- **Text-to-Speech (TTS)**: Convert responses to speech
- **Two Engines**:
  - **gTTS**: Google TTS (high quality, online)
  - **pyttsx3**: Offline TTS (works without internet)
- **Audio Streaming**: Play responses directly in browser
- **Customizable**: Adjustable rate, volume, voice

### Usage
1. Enable "Voice Input/Output" in sidebar
2. Click 🎤 button to speak
3. Click 🔊 to hear response

## NLP Analysis

### Sentiment Analysis
- **Polarity Detection**: -1 (negative) to +1 (positive)
- **Subjectivity Scoring**: 0 (objective) to 1 (subjective)
- **Conversation Analysis**: Track sentiment over entire chat
- **Per-Message Analysis**: Analyze individual messages

### Entity Recognition
- **Named Entity Recognition (NER)**: Extract entities from text
- **Entity Types**:
  - Persons (PERSON)
  - Organizations (ORG)
  - Locations (GPE, LOC)
  - Dates (DATE)
  - Products, Events, and more

### Keyword Extraction
- **Automatic Keyword Detection**: Identify important terms
- **Relevance Scoring**: Ranked by importance
- **Noun Chunk Extraction**: Meaningful phrases

### Usage
1. Enable "NLP Analysis" in sidebar
2. View analytics in "📊 Analytics" tab
3. See sentiment trends and entity summaries

## Multi-language Support

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)
- Portuguese (pt)
- Russian (ru)
- Italian (it)
- Dutch (nl)
- Polish (pl)
- Turkish (tr)

### Features
- **UI Translation**: Interface in 15+ languages
- **Language Detection**: Automatic detection of input language
- **Multilingual Responses**: Instruct AI to respond in specific language
- **Native Language Names**: UI shows languages in their native script

### Usage
Select language from dropdown in sidebar. The UI will update immediately.

## Logging & Monitoring

### Logging Features
- **Structured Logging**: Using loguru for better log management
- **Log Rotation**: Automatic rotation at 10 MB
- **Log Retention**: Keep logs for 7 days
- **Contextual Logging**: Logs include metadata

### Metrics Tracking
- **Message Count**: Track total messages
- **Token Usage**: Monitor token consumption
- **Cost Tracking**: Real-time cost calculation
- **Response Times**: Track AI response latency
- **Provider Usage**: See which providers are used most
- **Error Tracking**: Monitor and count errors

### Prometheus Integration (Optional)
- **Metrics Export**: Export to Prometheus for monitoring
- **Dashboards**: Create Grafana dashboards
- **Alerting**: Set up alerts based on metrics

### Session Metrics
View real-time metrics in sidebar:
- Total messages sent
- Total tokens used
- Total cost
- Average response time

## AI Agents

### Available Agents

#### 1. Research Agent
- **Purpose**: Research information from documents
- **Use Case**: Find specific information in uploaded PDFs, texts, etc.
- **Features**: Uses vector search for accurate retrieval

#### 2. Summarizer Agent
- **Purpose**: Create concise summaries of long text
- **Use Case**: Summarize articles, documents, or chat history
- **Features**: Configurable summary length

#### 3. Question Answering Agent
- **Purpose**: Answer questions using RAG (Retrieval-Augmented Generation)
- **Use Case**: Ask questions about uploaded documents
- **Features**: Combines retrieval with AI generation

#### 4. Data Analysis Agent
- **Purpose**: Analyze data and generate insights
- **Use Case**: Analyze CSV files, JSON data
- **Features**: Statistical analysis, pattern detection

### Agent Orchestration
- **Pipeline Execution**: Chain multiple agents together
- **Task Queuing**: Queue tasks for sequential execution
- **Status Tracking**: Monitor agent execution status
- **Error Handling**: Graceful failure handling

### Usage
1. Enable "AI Agents" in sidebar
2. Navigate to "🤖 Agents" tab
3. Select agent type and configure parameters
4. Execute task and view results

## Enhanced RAG

### Text Splitting Strategies

#### 1. Recursive Character Splitter (Default)
- Splits on multiple separators hierarchically
- Best for: General text documents

#### 2. Token-based Splitter
- Splits based on actual token count
- Best for: Precise token management

#### 3. Character Splitter
- Splits on specific characters
- Best for: Simple documents

#### 4. Markdown Splitter
- Preserves markdown structure
- Best for: Markdown documents

#### 5. Code Splitter
- Aware of code syntax
- Best for: Python code files

### Advanced Search Methods

#### 1. Similarity Search
- Standard vector similarity
- Fast and accurate

#### 2. MMR (Maximal Marginal Relevance)
- Balances relevance and diversity
- Reduces redundancy in results

#### 3. Hybrid Search
- Combines semantic and keyword search
- Best overall accuracy

### Enhanced Features
- **Metadata Filtering**: Filter results by document type, source, etc.
- **Relevance Scoring**: All results include confidence scores
- **Context Formatting**: Smart formatting for RAG prompts
- **Token-Aware Context**: Respects token limits
- **Persistent Storage**: Optional persistent vector DB
- **Custom Embeddings**: Use different embedding models

### Configuration
Adjust RAG settings in sidebar:
- **Chunk Size**: Size of document chunks (100-2000 characters)
- **Chunk Overlap**: Overlap between chunks (0-500 characters)
- **Number of Results**: How many chunks to retrieve (1-20)
- **Search Type**: similarity, mmr, or hybrid

## Getting Started

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Optional dependencies** (for full features):
```bash
# For voice features
pip install SpeechRecognition pyttsx3 gTTS pyaudio

# For NLP features
pip install spacy textblob
python -m spacy download en_core_web_sm

# For database features
pip install pymongo psycopg2-binary firebase-admin
```

3. **Set up API keys**:
```bash
export OPENAI=your_openai_key
export ANTHROPIC=your_anthropic_key
export MISTRAL=your_mistral_key
export COHERE=your_cohere_key
export GOOGLE_API_KEY=your_google_key
```

### Running the App

**Standard version**:
```bash
streamlit run main.py
```

**Enhanced version** (with all features):
```bash
streamlit run main_enhanced.py
```

## Environment Variables

### Required
- `OPENAI`: OpenAI API key
- `ANTHROPIC`: Anthropic API key (for Claude)
- `MISTRAL`: MistralAI API key
- `COHERE`: Cohere API key
- `GOOGLE_API_KEY`: Google AI API key (for Gemini)

### Optional
- `DB_TYPE`: Database type (json, mongodb, postgresql, firebase)
- `MONGODB_URI`: MongoDB connection string
- `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: PostgreSQL config
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Troubleshooting

### Voice features not working
- Ensure microphone permissions are granted
- Install PyAudio: `pip install pyaudio`
- For Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`
- For macOS: `brew install portaudio`

### NLP features not available
- Install spaCy model: `python -m spacy download en_core_web_sm`
- Install textblob: `pip install textblob`

### Database connection fails
- Check connection parameters
- Ensure database server is running
- Verify network connectivity
- Check firewall settings

### Google (Gemini) not working
- Verify `GOOGLE_API_KEY` is set correctly
- Check API key has proper permissions
- Ensure billing is enabled on Google Cloud

## Performance Tips

1. **Use GPT-4o-mini or Claude 3 Haiku** for faster, cheaper responses
2. **Adjust chunk size** based on document type
3. **Use hybrid search** for best RAG accuracy
4. **Enable Prometheus** for production monitoring
5. **Use PostgreSQL or MongoDB** for better performance at scale
6. **Limit max_tokens** for faster responses
7. **Use MMR search** to reduce redundant results

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Database**: Use strong passwords for database connections
3. **Secrets**: Use Kubernetes secrets or environment variables
4. **Sanitization**: User inputs are validated via Pydantic
5. **Rate Limiting**: Implement rate limiting for production use

## License

MIT License - See LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [your-repo/issues](https://github.com/your-repo/issues)
- Documentation: [your-docs-site](https://your-docs-site.com)
