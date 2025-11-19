"""Modernized main.py using modular architecture with Pydantic validation."""
import os
import sys
import streamlit as st
import json
from typing import List, Optional

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

# Streamlit configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Initialize configuration manager
@st.cache_resource
def get_config_manager():
    return ConfigManager()

config_manager = get_config_manager()

# Sidebar setup
st.sidebar.image("Logo.png", use_column_width=True)
st.title("Multiple AI ChatBot - OpenAI, MistralAI, Anthropic and Cohere")
st.subheader("Chat with the AI models from OpenAI, MistralAI, Anthropic and Cohere. Select the AI provider and the model to chat.")
st.sidebar.title("Settings")
st.sidebar.subheader("Select AI Provider")

try:
    # AI Provider Selection
    ai_providers = ["OpenAI", "MistralAI", "Anthropic", "Cohere"]
    ai_provider = st.sidebar.radio("Choose AI Provider", ai_providers)

    # Model selection based on the provider
    model_options = {
        "OpenAI": ['gpt-4', "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct"],
        "MistralAI": ['open-mistral-7b', 'open-mixtral-8x7b', 'open-mixtral-8x22b', 'mistral-small-latest', 'mistral-medium-latest', 'mistral-large-latest', 'mistral-embed'],
        "Anthropic": ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-2.1', 'claude-2.0', 'claude-instant-1.2'],
        "Cohere": ['command-r-plus', 'command-r', 'command', 'command-nightly', 'command-light', 'command-light-nightly']
    }
    
    model = st.sidebar.selectbox("Select Model", model_options[ai_provider])

    # Model settings
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 4000, 1000, 100)

    # Create provider configuration
    provider_config = AIProviderConfig(
        provider=ai_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # System prompts
    st.sidebar.subheader("System Prompts")
    system_prompts = {
        "Default": "You are a helpful assistant.",
        "Code Assistant": "You are a helpful coding assistant. Provide clear, accurate code examples and explanations.",
        "Document Analyzer": "You are an expert at analyzing documents. Provide detailed and accurate analysis.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling and creative content.",
        "Technical Explainer": "You are a technical explainer. Break down complex concepts into simple terms."
    }
    
    selected_prompt = st.sidebar.selectbox("Choose System Prompt", list(system_prompts.keys()))
    pre_prompt = system_prompts[selected_prompt]

    # Document upload section
    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'txt', 'csv'])
    url = st.sidebar.text_input("Or enter a URL:")

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

    # Chat state management
    st.sidebar.subheader("Chat State")
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

    # Add system message if it doesn't exist
    if st.session_state.messages and st.session_state.messages[0].get("role") != "system":
        system_message = ChatMessage(role="system", content=pre_prompt)
        st.session_state.messages.insert(0, {"role": "system", "content": pre_prompt})
    elif not st.session_state.messages:
        system_message = ChatMessage(role="system", content=pre_prompt)
        st.session_state.messages.append({"role": "system", "content": pre_prompt})

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is up?"):
        # Add user message to session state
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Enhance prompt with document search if applicable
        enhanced_prompt = pre_prompt
        try:
            if url:
                search_results = perform_vector_db_search("web", prompt, config_manager, url=url)
                enhanced_prompt += " " + str([result.content for result in search_results])
            elif text_file:
                search_results = perform_vector_db_search("text", prompt, config_manager)
                enhanced_prompt += " " + str([result.content for result in search_results])
            elif csv_file:
                search_results = perform_vector_db_search("csv", prompt, config_manager)
                enhanced_prompt += " " + str([result.content for result in search_results])
            elif pdf_file:
                search_results = perform_vector_db_search("pdf", prompt, config_manager)
                enhanced_prompt += " " + str([result.content for result in search_results])
        except Exception as e:
            st.error(f"Error processing document: {e}")

        # Update system message with enhanced context if document search was performed
        if enhanced_prompt != pre_prompt:
            # Update the system message in session state
            st.session_state.messages[0]["content"] = enhanced_prompt

        # Convert messages to Pydantic models
        chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]

        # Generate AI response
        try:
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

            # Calculate and display costs
            input_tokens = sum(provider.count_tokens(msg.content) for msg in chat_messages)
            output_tokens = provider.count_tokens(full_response)
            cost = config_manager.calculate_cost(model, input_tokens, output_tokens)
            
            st.write(f"Approximate cost of the chat: ${cost:.6f}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please reload the page and try again.")

except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.write("Please check your configuration and reload the page.")