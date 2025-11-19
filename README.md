# AI ChatBot Pro - Enhanced Multi-Provider AI Assistant

**Now with 5 AI Providers + Advanced Features!**

CodeChatBot Pro is an advanced AI chatbot application that integrates multiple AI providers with cutting-edge features including voice I/O, NLP analysis, multi-language support, and intelligent document processing.

## 🌟 Latest Updates

### New AI Provider
- **Google Gemini**: Access to Gemini 1.5 Pro, Flash, and Pro models

### Latest Models (2024)
- **OpenAI**: GPT-4o, GPT-4o-mini, o1-preview, o1-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **MistralAI**: Mistral Large, Medium, Small, Mixtral models
- **Cohere**: Command R+, Command R
- **Google**: Gemini 1.5 Pro/Flash

### Enhanced Features
- 🎙️ **Voice Input/Output**: Speak to the AI and hear responses
- 📊 **NLP Analysis**: Sentiment analysis and entity recognition
- 🌍 **Multi-language Support**: 15+ languages
- 💾 **Database Integration**: PostgreSQL, MongoDB, Firebase, JSON
- 📈 **Monitoring & Logging**: Comprehensive metrics and logging
- 🤖 **AI Agents**: Autonomous task execution
- 🔍 **Enhanced RAG**: Multiple chunking strategies, hybrid search
- 🎨 **Modern UI**: Tabs, expandable sections, metrics dashboard

## Chatbot Interface
![Chatbot Interface](Preview.png)

## Core Features

### AI Capabilities
- ✅ Chat with 5 different AI providers (OpenAI, Anthropic, MistralAI, Cohere, Google)
- ✅ 25+ different AI models to choose from
- ✅ Switch providers and models mid-conversation
- ✅ Adjustable temperature and token limits
- ✅ Real-time cost tracking

### Document Processing
- ✅ Upload and analyze PDFs, TXT, CSV files
- ✅ Process web pages via URL
- ✅ Advanced RAG with multiple chunking strategies
- ✅ Hybrid search (semantic + keyword)
- ✅ Metadata filtering and relevance scoring

### Advanced Features
- 🎙️ **Voice Interaction**: Speech-to-text and text-to-speech
- 📊 **Sentiment Analysis**: Track conversation sentiment
- 🏷️ **Entity Recognition**: Extract people, places, organizations
- 🌐 **15+ Languages**: Full UI translation
- 💾 **Persistent Storage**: Multiple database backends
- 📈 **Analytics Dashboard**: Metrics, costs, token usage
- 🤖 **AI Agents**: Research, summarization, Q&A agents

## Quick Start

### Installation

**Option 1: Modern Installation with uv (Recommended)**
```bash
git clone https://github.com/CodeHalwell/code_chat_bot.git
cd code_chat_bot

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment variables
touch .env
echo OPENAI=your_openai_key >> .env
echo MISTRAL=your_mistral_key >> .env
echo ANTHROPIC=your_anthropic_key >> .env
echo COHERE=your_cohere_key >> .env
echo GOOGLE_API_KEY=your_google_key >> .env

# Run the standard app
uv run streamlit run main.py

# Or run the enhanced app (with all features)
uv run streamlit run main_enhanced.py
```

**Option 2: pip Installation**
```bash
git clone https://github.com/CodeHalwell/code_chat_bot.git
cd code_chat_bot

# Install core dependencies
pip install -r requirements.txt

# Set up environment variables
touch .env
echo OPENAI=your_openai_key >> .env
echo MISTRAL=your_mistral_key >> .env
echo ANTHROPIC=your_anthropic_key >> .env
echo COHERE=your_cohere_key >> .env
echo GOOGLE_API_KEY=your_google_key >> .env

# Run the app
streamlit run main.py
```

### Optional Feature Installation

**For Voice Features:**
```bash
pip install SpeechRecognition pyttsx3 gTTS pyaudio
# Linux: sudo apt-get install portaudio19-dev python3-pyaudio
# macOS: brew install portaudio
```

**For NLP Features:**
```bash
pip install spacy textblob
python -m spacy download en_core_web_sm
```

**For Database Features:**
```bash
# PostgreSQL
pip install psycopg2-binary

# MongoDB
pip install pymongo

# Firebase
pip install firebase-admin
```

## 📁 Project Structure

```
code_chat_bot/
├── main.py                    # Standard Streamlit app
├── main_enhanced.py           # Enhanced app with all features
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Modern package configuration
├── FEATURES.md               # Detailed features documentation
├── .env                      # API keys (create this)
├── Logo.png                  # App logo
├── Preview.png               # Screenshot
├── src/code_chat_bot/        # Modular architecture
│   ├── __init__.py          # Main exports
│   ├── config/              # Configuration management
│   ├── models/              # Pydantic data models
│   ├── providers/           # AI provider implementations
│   ├── document_processing/ # RAG and document handling
│   ├── database/            # Database integrations
│   ├── voice/               # Voice input/output
│   ├── nlp_analysis/        # NLP features
│   ├── i18n/                # Multi-language support
│   ├── monitoring/          # Logging and metrics
│   └── agents/              # AI agents
├── previous_chats/          # Saved chat logs
└── upload_docs/             # Document staging area
```

## 🚀 How to Use

### Basic Usage
1. Start the chatbot: `streamlit run main.py`
2. Select an AI provider (OpenAI, Anthropic, etc.)
3. Choose a model from the dropdown
4. Adjust temperature and max tokens if needed
5. Select a system prompt for your use case
6. Start chatting!

### Advanced Features

**Document Processing:**
1. Upload a file (PDF, TXT, CSV) OR enter a URL
2. Configure RAG settings (chunk size, search type)
3. Ask questions about your document
4. The AI will use relevant context to answer

**Voice Interaction:**
1. Enable "Voice Input/Output" in sidebar
2. Click 🎤 to speak your question
3. Click 🔊 to hear the AI's response

**Analytics:**
1. Switch to "📊 Analytics" tab
2. View sentiment analysis of conversation
3. See extracted entities and keywords
4. Monitor costs and token usage

**AI Agents:**
1. Enable "AI Agents" in sidebar
2. Navigate to "🤖 Agents" tab
3. Choose an agent (Summarizer, Research, Q&A)
4. Configure parameters and run

**Multi-language:**
1. Select your language from dropdown
2. UI updates automatically
3. AI can respond in your chosen language

**Database Persistence:**
1. Enter a session ID
2. Click "💾 Save" to persist chat
3. Click "📂 Load" to restore later
4. View recent sessions in dropdown

## ⚙️ Configuration

### Environment Variables

**Required:**
- `OPENAI`: Your OpenAI API key
- `ANTHROPIC`: Your Anthropic API key
- `MISTRAL`: Your MistralAI API key
- `COHERE`: Your Cohere API key
- `GOOGLE_API_KEY`: Your Google AI API key

**Optional:**
- `DB_TYPE`: Database type (json, mongodb, postgresql, firebase)
- `MONGODB_URI`: MongoDB connection string
- `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Database Setup

**PostgreSQL:**
```bash
export DB_TYPE=postgresql
export POSTGRES_HOST=localhost
export POSTGRES_DB=chatbot
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
```

**MongoDB:**
```bash
export DB_TYPE=mongodb
export MONGODB_URI=mongodb://localhost:27017/
```

## 📚 Documentation

- **[FEATURES.md](FEATURES.md)**: Comprehensive feature documentation
- **[MODERNIZATION.md](MODERNIZATION.md)**: Architecture details

## 🤝 How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [OpenAI](https://openai.com) - GPT models
- [Anthropic](https://anthropic.com) - Claude models
- [MistralAI](https://mistral.ai) - Mistral models
- [Cohere](https://cohere.ai) - Command models
- [Google AI](https://ai.google.dev/) - Gemini models
- [Streamlit](https://streamlit.io) - Web framework
- [LangChain](https://langchain.com) - RAG framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ❤️ by CodeHalwell**





