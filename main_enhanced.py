"""Enhanced main.py with all latest features and improvements."""
import os
import sys
import streamlit as st
import json
from typing import List, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from code_chat_bot import (
    ConfigManager,
    ChatMessage,
    AIProviderConfig,
    DocumentMetadata,
    get_provider,
    perform_vector_db_search
)

# Import new modules
try:
    from code_chat_bot.database import get_database_provider
    from code_chat_bot.voice import VoiceManager
    from code_chat_bot.nlp_analysis import NLPAnalyzer
    from code_chat_bot.i18n import I18nManager, SUPPORTED_LANGUAGES
    from code_chat_bot.monitoring import MonitoringManager
    from code_chat_bot.agents import AgentOrchestrator, create_research_agent, create_summarizer_agent
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"Some enhanced features not available: {e}")

# Streamlit configuration with theme
st.set_page_config(
    page_title="AI ChatBot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI ChatBot Pro\nMultiple AI providers with advanced features"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize managers
@st.cache_resource
def get_config_manager():
    return ConfigManager()

@st.cache_resource
def get_i18n_manager():
    if ENHANCED_FEATURES_AVAILABLE:
        return I18nManager(default_language="en")
    return None

@st.cache_resource
def get_monitoring_manager():
    if ENHANCED_FEATURES_AVAILABLE:
        return MonitoringManager(log_file="chatbot.log", enable_prometheus=False)
    return None

@st.cache_resource
def get_voice_manager():
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            return VoiceManager(language="en")
        except:
            return None
    return None

@st.cache_resource
def get_nlp_analyzer():
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            return NLPAnalyzer()
        except:
            return None
    return None

config_manager = get_config_manager()
i18n = get_i18n_manager()
monitoring = get_monitoring_manager()
voice_manager = get_voice_manager()
nlp_analyzer = get_nlp_analyzer()

# Initialize database provider
@st.cache_resource
def get_db_provider():
    if ENHANCED_FEATURES_AVAILABLE:
        try:
            # Default to JSON, can be configured via environment
            db_type = os.getenv("DB_TYPE", "json")
            if db_type == "mongodb":
                return get_database_provider(
                    "mongodb",
                    connection_string=os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
                )
            elif db_type == "postgresql":
                return get_database_provider(
                    "postgresql",
                    connection_params={
                        "host": os.getenv("POSTGRES_HOST", "localhost"),
                        "database": os.getenv("POSTGRES_DB", "chatbot"),
                        "user": os.getenv("POSTGRES_USER", "postgres"),
                        "password": os.getenv("POSTGRES_PASSWORD", "password")
                    }
                )
            else:
                return get_database_provider("json", data_dir="previous_chats")
        except:
            return None
    return None

db_provider = get_db_provider()

# Sidebar setup
st.sidebar.image("Logo.png", use_column_width=True)

# Language selector
if i18n:
    selected_lang = st.sidebar.selectbox(
        "Language / Idioma / 语言",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: f"{SUPPORTED_LANGUAGES[x].native_name} ({SUPPORTED_LANGUAGES[x].name})"
    )
    i18n.set_language(selected_lang)
    title = i18n.t("app_title")
else:
    title = "AI ChatBot Pro"
    selected_lang = "en"

st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
st.subheader("Chat with the latest AI models from OpenAI, MistralAI, Anthropic, Cohere, and Google")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
st.sidebar.subheader("🤖 AI Provider")

try:
    # AI Provider Selection - now includes Google
    ai_providers = ["OpenAI", "MistralAI", "Anthropic", "Cohere", "Google"]
    ai_provider = st.sidebar.radio("Choose AI Provider", ai_providers)

    # Model selection based on provider with latest models
    model_options = {
        "OpenAI": [
            'gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini',
            'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
        ],
        "MistralAI": [
            'mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest',
            'open-mixtral-8x22b', 'open-mixtral-8x7b', 'open-mistral-7b', 'mistral-embed'
        ],
        "Anthropic": [
            'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229',
            'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'
        ],
        "Cohere": [
            'command-r-plus', 'command-r', 'command', 'command-light'
        ],
        "Google": [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'
        ]
    }

    model = st.sidebar.selectbox("Select Model", model_options[ai_provider])

    # Model settings
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 8000, 2000, 100)

    # Create provider configuration
    provider_config = AIProviderConfig(
        provider=ai_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # System prompts with more options
    st.sidebar.subheader("📝 System Prompts")
    system_prompts = {
        "Default": "You are a helpful assistant.",
        "Code Assistant": "You are a helpful coding assistant. Provide clear, accurate code examples and explanations.",
        "Document Analyzer": "You are an expert at analyzing documents. Provide detailed and accurate analysis.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling and creative content.",
        "Technical Explainer": "You are a technical explainer. Break down complex concepts into simple terms.",
        "Data Analyst": "You are a data analysis expert. Help analyze data and generate insights.",
        "Multilingual": f"You are a multilingual assistant. Respond in {SUPPORTED_LANGUAGES.get(selected_lang).name}."
    }

    selected_prompt = st.sidebar.selectbox("Choose System Prompt", list(system_prompts.keys()))
    pre_prompt = system_prompts[selected_prompt]

    # Document upload section with enhanced options
    st.sidebar.subheader("📄 Document Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'txt', 'csv'])
    url = st.sidebar.text_input("Or enter a URL:")

    # RAG Settings
    with st.sidebar.expander("🔍 RAG Settings"):
        chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 75, 25)
        search_k = st.slider("Number of Results", 1, 20, 10, 1)
        search_type = st.selectbox("Search Type", ["similarity", "mmr", "hybrid"])

    # Process uploaded files
    pdf_file = text_file = csv_file = None
    if uploaded_file is not None:
        file_type = uploaded_file.type
        if 'pdf' in file_type:
            pdf_file = uploaded_file
            with open('./upload_docs/file.pdf', 'wb') as f:
                f.write(uploaded_file.read())
        elif 'text' in file_type:
            text_file = uploaded_file
            with open('./upload_docs/file.txt', 'wb') as f:
                f.write(uploaded_file.read())
        elif 'csv' in file_type:
            csv_file = uploaded_file
            with open('./upload_docs/file.csv', 'wb') as f:
                f.write(uploaded_file.read())

    # Enhanced Features Toggle
    st.sidebar.subheader("✨ Enhanced Features")
    enable_voice = st.sidebar.checkbox("Enable Voice Input/Output", value=False)
    enable_nlp = st.sidebar.checkbox("Enable NLP Analysis", value=False)
    enable_agents = st.sidebar.checkbox("Enable AI Agents", value=False)

    # Chat State Management with Database
    st.sidebar.subheader("💾 Chat Management")

    if db_provider:
        save_session_id = st.sidebar.text_input("Session ID:", f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if 'messages' in st.session_state:
                    metadata = {
                        "provider": ai_provider,
                        "model": model,
                        "timestamp": datetime.now().isoformat()
                    }
                    chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                    success = db_provider.save_chat_session(save_session_id, chat_messages, metadata)
                    if success:
                        st.sidebar.success("Chat saved!")
                    else:
                        st.sidebar.error("Failed to save")

        with col2:
            if st.button("📂 Load", use_container_width=True):
                session = db_provider.load_chat_session(save_session_id)
                if session:
                    st.session_state.messages = session['messages']
                    st.sidebar.success("Chat loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("Session not found")

        # List recent sessions
        with st.sidebar.expander("Recent Sessions"):
            sessions = db_provider.list_chat_sessions(limit=10)
            for session in sessions:
                st.write(f"- {session['session_id']} ({session.get('message_count', 0)} msgs)")
    else:
        # Fallback to JSON
        save_filename = st.sidebar.text_input("Save filename:", "chat_log")
        if st.sidebar.button("Save Chat"):
            if 'messages' in st.session_state:
                with open(f'./previous_chats/{save_filename}.json', 'w') as f:
                    json.dump(st.session_state.messages, f)
                st.sidebar.success(f"Chat saved as {save_filename}.json")

        load_filename = st.sidebar.text_input("Load filename:")
        if st.sidebar.button("Load Chat") and load_filename:
            try:
                with open(f'./previous_chats/{load_filename}.json', 'r') as f:
                    st.session_state.messages = json.load(f)
                st.sidebar.success(f"Chat loaded from {load_filename}.json")
                st.rerun()
            except FileNotFoundError:
                st.sidebar.error("File not found")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    # Add system message if it doesn't exist
    if st.session_state.messages and st.session_state.messages[0].get("role") != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": pre_prompt})
    elif not st.session_state.messages:
        st.session_state.messages.append({"role": "system", "content": pre_prompt})

    # Main chat interface with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🎙️ Voice", "🤖 Agents"])

    with tab1:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Voice input button
        if enable_voice and voice_manager and voice_manager.input_available:
            if st.button("🎤 Voice Input"):
                with st.spinner("Listening..."):
                    voice_text = voice_manager.listen(timeout=10, phrase_time_limit=15)
                    if voice_text:
                        st.success(f"You said: {voice_text}")
                        # Set the voice text as input
                        st.session_state.voice_input = voice_text

        # Chat input
        user_input = st.chat_input("What is up?")

        # Use voice input if available
        if hasattr(st.session_state, 'voice_input'):
            user_input = st.session_state.voice_input
            del st.session_state.voice_input

        if user_input:
            # Add user message to session state
            user_message = {"role": "user", "content": user_input}
            st.session_state.messages.append(user_message)

            with st.chat_message("user"):
                st.markdown(user_input)

            # Enhance prompt with document search if applicable
            enhanced_prompt = pre_prompt
            search_results = []

            try:
                if url:
                    search_results = perform_vector_db_search("web", user_input, config_manager, url=url, k=search_k)
                    enhanced_prompt += "\n\nRelevant context from web page:\n" + "\n".join([f"{i+1}. {r.content}" for i, r in enumerate(search_results)])
                elif text_file:
                    search_results = perform_vector_db_search("text", user_input, config_manager, k=search_k)
                    enhanced_prompt += "\n\nRelevant context from document:\n" + "\n".join([f"{i+1}. {r.content}" for i, r in enumerate(search_results)])
                elif csv_file:
                    search_results = perform_vector_db_search("csv", user_input, config_manager, k=search_k)
                    enhanced_prompt += "\n\nRelevant context from CSV:\n" + "\n".join([f"{i+1}. {r.content}" for i, r in enumerate(search_results)])
                elif pdf_file:
                    search_results = perform_vector_db_search("pdf", user_input, config_manager, k=search_k)
                    enhanced_prompt += "\n\nRelevant context from PDF:\n" + "\n".join([f"{i+1}. {r.content}" for i, r in enumerate(search_results)])
            except Exception as e:
                st.error(f"Error processing document: {e}")

            # Update system message with enhanced context
            if enhanced_prompt != pre_prompt:
                st.session_state.messages[0]["content"] = enhanced_prompt

            # Convert messages to Pydantic models
            chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]

            # Generate AI response
            try:
                import time
                start_time = time.time()

                provider = get_provider(provider_config, config_manager)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    # Stream response
                    for chunk in provider.generate_response(chat_messages, stream=True):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # Add assistant response to session state
                    assistant_message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(assistant_message)

                response_time = time.time() - start_time

                # Calculate and display costs
                input_tokens = sum(provider.count_tokens(msg.content) for msg in chat_messages)
                output_tokens = provider.count_tokens(full_response)
                cost = config_manager.calculate_cost(model, input_tokens, output_tokens)
                st.session_state.total_cost += cost

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cost", f"${cost:.6f}")
                with col2:
                    st.metric("Input Tokens", f"{input_tokens:,}")
                with col3:
                    st.metric("Output Tokens", f"{output_tokens:,}")
                with col4:
                    st.metric("Response Time", f"{response_time:.2f}s")

                # Log to monitoring system
                if monitoring:
                    monitoring.log_and_record_message(
                        provider=ai_provider,
                        model=model,
                        role="assistant",
                        tokens=input_tokens + output_tokens,
                        cost=cost,
                        response_time=response_time
                    )

                # Voice output
                if enable_voice and voice_manager and voice_manager.output_available:
                    if st.button("🔊 Read Response"):
                        audio_bytes = voice_manager.get_audio_bytes(full_response[:500])  # Limit length
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                if monitoring:
                    monitoring.log_and_record_error(e, error_type=type(e).__name__)

    with tab2:
        st.subheader("📊 Chat Analytics")

        if len(st.session_state.messages) > 1:
            # Cost summary
            st.metric("Total Session Cost", f"${st.session_state.total_cost:.6f}")

            # NLP Analysis
            if enable_nlp and nlp_analyzer and nlp_analyzer.sentiment_available:
                st.subheader("💭 Sentiment Analysis")

                user_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
                if user_messages:
                    sentiment_summary = nlp_analyzer.get_sentiment_summary(user_messages)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive", sentiment_summary.get("positive", 0))
                    with col2:
                        st.metric("Neutral", sentiment_summary.get("neutral", 0))
                    with col3:
                        st.metric("Negative", sentiment_summary.get("negative", 0))

                    st.write(f"**Overall Sentiment:** {sentiment_summary.get('overall_sentiment', 'neutral').title()}")
                    st.write(f"**Average Polarity:** {sentiment_summary.get('avg_polarity', 0):.2f}")

            # Entity Recognition
            if enable_nlp and nlp_analyzer and nlp_analyzer.entity_recognition_available:
                st.subheader("🏷️ Entity Recognition")

                all_text = " ".join([msg["content"] for msg in st.session_state.messages if msg["role"] != "system"])
                entity_summary = nlp_analyzer.get_entity_summary(all_text)

                if entity_summary and "error" not in entity_summary:
                    for entity_type, entities in entity_summary.items():
                        with st.expander(f"{entity_type} ({len(entities)})"):
                            st.write(", ".join(entities[:10]))  # Show first 10
        else:
            st.info("Start chatting to see analytics!")

    with tab3:
        st.subheader("🎙️ Voice Features")

        if voice_manager:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Voice Input**")
                if voice_manager.input_available:
                    st.success("✅ Available")
                    if st.button("Test Microphone"):
                        with st.spinner("Listening for 5 seconds..."):
                            text = voice_manager.listen(timeout=5)
                            if text:
                                st.write(f"Recognized: {text}")
                            else:
                                st.warning("No speech detected")
                else:
                    st.error("❌ Not available")

            with col2:
                st.write("**Voice Output**")
                if voice_manager.output_available:
                    st.success("✅ Available")
                    test_text = st.text_input("Test text-to-speech:", "Hello, this is a test")
                    if st.button("Test Speaker"):
                        audio_bytes = voice_manager.get_audio_bytes(test_text)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.error("❌ Not available")
        else:
            st.warning("Voice features not available. Install required dependencies.")

    with tab4:
        st.subheader("🤖 AI Agents")

        if enable_agents and ENHANCED_FEATURES_AVAILABLE:
            st.write("AI Agents can perform specialized tasks autonomously.")

            agent_type = st.selectbox("Select Agent", [
                "Research Agent", "Summarizer Agent", "Question Answering Agent"
            ])

            if agent_type == "Summarizer Agent":
                text_to_summarize = st.text_area("Text to summarize:", height=200)
                max_length = st.slider("Max summary length:", 50, 500, 200)

                if st.button("Summarize"):
                    from code_chat_bot.agents import create_summarizer_agent, AgentTask
                    agent = create_summarizer_agent()
                    task = AgentTask(
                        name="summarize_text",
                        description="Summarize the provided text",
                        parameters={"text": text_to_summarize, "max_length": max_length}
                    )
                    result = agent.run(task)

                    if result.success:
                        st.success("Summary:")
                        st.write(result.output)
                    else:
                        st.error(f"Error: {result.error}")
        else:
            st.info("Enable AI Agents in the sidebar to use this feature.")

    # Monitoring dashboard in sidebar
    if monitoring:
        with st.sidebar.expander("📈 Session Metrics"):
            metrics = monitoring.metrics.get_summary()
            st.write(f"**Messages:** {metrics['total_messages']}")
            st.write(f"**Total Tokens:** {metrics['total_tokens']:,}")
            st.write(f"**Total Cost:** ${metrics['total_cost']:.6f}")
            if metrics['avg_response_time'] > 0:
                st.write(f"**Avg Response:** {metrics['avg_response_time']:.2f}s")

except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.write("Please check your configuration and reload the page.")
    if monitoring:
        monitoring.log_and_record_error(e, error_type="initialization_error")
