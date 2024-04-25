import os
import dotenv
from openai import OpenAI
from mistralai.client import MistralClient
from anthropic import Anthropic
import streamlit as st
import json
import tiktoken
import cohere
from document_loader import DocumentLoader


#####################################Streamlit############################################

# default the streamlit app to dark mode
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


#####################################Model Dictionary & Cost Calculation############################################

cost = {
    "open-mistral-7b": {
        "description": "A 7B transformer model, fast-deployed and easily customizable for various applications.",
        "input_price_1M_tokens": 0.25,
        "output_price_1M_tokens": 0.25
    },
    "open-mixtral-8x7b": {
        "description": "A 7B sparse Mixture-of-Experts model with 12.9B active parameters from a total of 45B, designed for efficient large-scale processing.",
        "input_price_1M_tokens": 0.7,
        "output_price_1M_tokens": 0.7
    },
    "open-mixtral-8x22b": {
        "description": "A high-performance 22B sparse Mixture-of-Experts model utilizing 39B active parameters from 141B total, suitable for complex problem solving.",
        "input_price_1M_tokens": 2,
        "output_price_1M_tokens": 6
    },
    "mistral-small-latest": {
        "description": "Designed for cost-effective reasoning with low latency, ideal for quick response applications.",
        "input_price_1M_tokens": 2,
        "output_price_1M_tokens": 6
    },
    "mistral-medium-latest": {
        "description": "Medium-scale model providing a balance between performance and cost, suitable for a range of applications.",
        "input_price_1M_tokens": 2.7,
        "output_price_1M_tokens": 8.1
    },
    "mistral-large-latest": {
        "description": "The flagship model of the Mistral series, offering advanced reasoning capabilities for the most demanding tasks.",
        "input_price_1M_tokens": 8,
        "output_price_1M_tokens": 24
    },
    "mistral-embed": {
        "description": "Advanced model for semantic extraction from text, ideal for creating meaningful text representations.",
        "input_price_1M_tokens": 0.1,
        "output_price_1M_tokens": 0.1
    },
    "claude-3-haiku-20240307": {
        "description": "Optimized for speed and efficiency, well-suited for lightweight tasks requiring quick turnarounds.",
        "input_price_1M_tokens": 0.25,
        "output_price_1M_tokens": 1.25
    },
    "claude-3-sonnet-20240229": {
        "description": "Designed for robust performance on demanding tasks, offering detailed and extensive responses.",
        "input_price_1M_tokens": 3,
        "output_price_1M_tokens": 15
    },
    "claude-3-opus-20240229": {
        "description": "The most advanced model in the Claude 3 series, engineered for superior performance on complex challenges.",
        "input_price_1M_tokens": 15,
        "output_price_1M_tokens": 75
    },
    "claude-2.1": {
        "description": "Features a 200K token context window and enhanced accuracy, with reduced model hallucination and new beta features for enterprise applications.",
        "input_price_1M_tokens": 8,
        "output_price_1M_tokens": 24
    },
    "claude-2.0": {
        "description": "Improved user interaction with extended memory and reduced harmful outputs, accessible via API and a new public interface.",
        "input_price_1M_tokens": 8,
        "output_price_1M_tokens": 24
    },
    "claude-instant-1.2": {
        "description": "A cost-effective model capable of handling casual dialogue, text analysis, summarization, and comprehension with quick response times.",
        "input_price_1M_tokens": 0.8,
        "output_price_1M_tokens": 2.4
    },
    "gpt-4": {
        "description": "OpenAI's GPT-4 offers transformative capabilities with 175B parameters, designed for a wide range of high-complexity tasks.",
        "input_price_1M_tokens": 30,
        "output_price_1M_tokens": 60
    },
    "gpt-4-turbo-2024-04-09": {
        "description": "A more efficient version of GPT-4, offering faster responses and reduced costs without compromising on quality.",
        "input_price_1M_tokens": 10,
        "output_price_1M_tokens": 30
    },
    "gpt-4-32k": {
        "description": "The high-end model of GPT-4 designed for tasks requiring extensive context handling and depth, with a 32k token limit.",
        "input_price_1M_tokens": 60,
        "output_price_1M_tokens": 120
    },
    "gpt-3.5-turbo-0125": {
        "description": "An optimized variant of GPT-3.5, offering high performance with a focus on speed and affordability.",
        "input_price_1M_tokens": 0.5,
        "output_price_1M_tokens": 1.5
    },
    "gpt-3.5-turbo-instuct": {
        "description": "Tailored for interactive applications, this model combines the capabilities of GPT-3.5 with enhanced directive compliance.",
        "input_price_1M_tokens": 1.5,
        "output_price_1M_tokens": 2
    },
    "text-embedding-3-large": {
        "description": "Specialized in creating dense vector representations from text, facilitating advanced machine learning applications.",
        "input_price_1M_tokens": 0.13,
        "output_price_1M_tokens": 0.14
    },
    "command-r-plus": {
        "description": "Cohere's advanced model with enhanced processing capabilities for complex natural language understanding tasks.",
        "input_price_1M_tokens": 3,
        "output_price_1M_tokens": 15
    },
    "command-r": {
        "description": "Provides robust language understanding with efficient processing, suitable for a variety of applications.",
        "input_price_1M_tokens": 0.5,
        "output_price_1M_tokens": 1.5
    },
    "command": {
        "description": "Entry-level model from Cohere, offering solid performance for general natural language processing tasks.",
        "input_price_1M_tokens": 0.5,
        "output_price_1M_tokens": 1.5
    },
    "embed-english-v3.0": {
        "description": "Advanced embedding capabilities for English text, ideal for applications requiring nuanced language understanding.",
        "input_price_1M_tokens": 0.1,
        "output_price_1M_tokens": 0.05
    }
}


#####################################Functions############################################

# calculate the cost of the chat
def calculate_cost(prompt, response, model):
    """
    Calculate the cost of the chat  based on the input and output tokens
    :param prompt:
    :param response:
    :param model:
    :return:
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Use a default encoding if the model is not found. Use the encoding for gpt-3.5-turbo-0125 as a rough estimate
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    input_tokens = len(encoding.encode(prompt))
    output_tokens = len(encoding.encode(response))
    input_cost = cost[model]["input_price_1M_tokens"] * input_tokens / 1_000_000
    output_cost = cost[model]["output_price_1M_tokens"] * output_tokens / 1_000_000
    return (input_cost + output_cost), input_tokens, output_tokens


# Function to clear session state
def clear_session():
    """
    Clear the session state of the app
    :return:
    """
    for key in st.session_state.keys():
        del st.session_state[key]

# Button to clear session state and rerun the app
if st.sidebar.button('Reset App', key='reset', help='Clear the session state and start over'):
    clear_session()
    st.rerun()


def save_state(filename):
    """
    Save the session state to a JSON file in the previous_chats folder
    :param filename:
    :return: JSON file
    """
    with open(f"./previous_chats/{filename}.json", 'w') as f:
        json.dump(dict(st.session_state), f)


def load_state(filename):
    """
    Load the session state from a JSON file in the previous_chats folder
    :param filename:
    :return: JSON file and rerun the app
    """
    with open(f"./previous_chats/{filename}.json", 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        st.session_state[key] = value
    st.rerun()


def display_state_as_markdown():
    """
    Display the session state as markdown
    :return: Returns the markdown text
    """
    markdown_text = "### Session State\n"
    for key, value in st.session_state.items():
        markdown_text += f"- **{key}**: {value}\n"
    st.markdown(markdown_text)


# add a running total cost of all messages by saving the total cost of each messsage in the session state
def add_cost_to_session_state(cost):
    """
    Add the cost of the chat to the session state
    :param cost:
    :return:
    """
    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0
    st.session_state["total_cost"] += cost


# open the total_cost file, read the number, add the cost of the chat to the total cost, and write the new total cost to the file
def add_cost_to_total_cost(cost):
    """
    Add the cost of the chat to the total cost file
    :param cost:
    :return:
    """
    with open("total_cost.txt", "r") as f:
        total_cost = float(f.read())
    total_cost += cost
    with open("total_cost.txt", "w") as f:
        f.write(str(total_cost))

# Load environment variables
dotenv.load_dotenv()

#####################################API Keys############################################

# API setup
openai_client = OpenAI(api_key=os.getenv("OPENAI"))
mistral_client = MistralClient(api_key=os.getenv("MISTRAL"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC"))
cohere_client = cohere.Client(api_key=os.getenv("COHERE"))


#####################################Streamlit UI############################################

st.sidebar.image("Logo.png", use_column_width=True)

# Streamlit UI setup
st.title("Multiple AI ChatBot - OpenAI, MistralAI, Anthropic and Cohere")
st.subheader("Chat with the AI models from OpenAI, MistralAI, Anthropic and Cohere. Select the AI provider and the model to chat.")
st.sidebar.title("Settings")
st.sidebar.subheader("Select AI Provider")

# Options for AI models from different providers
ai_providers = ["OpenAI", "MistralAI", "Anthropic", "Cohere"]
ai_provider = st.sidebar.radio("Choose AI Provider", ai_providers)


# Model selection based on the provider
if ai_provider == "OpenAI":
    openai_models = ['gpt-4', "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instuct", "text-embedding-3-large"]
    model = st.sidebar.selectbox("Select Model", openai_models)
elif ai_provider == "MistralAI":
    mistral_models = ['open-mistral-7b', 'open-mixtral-8x7b', 'open-mixtral-8x22b', 'mistral-small-latest',
                      'mistral-medium-latest', 'mistral-large-latest', 'mistral-embed']
    model = st.sidebar.selectbox("Select Model", mistral_models)
elif ai_provider == "Anthropic":
    anthropic_models = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-2.1',
                        'claude-2.0', 'claude-instant-1.2']
    model = st.sidebar.selectbox("Select Model", anthropic_models)
elif ai_provider == "Cohere":
    cohere_models = ['command-r-plus', "command-r", "command", "embed-english-v3.0"]
    model = st.sidebar.selectbox("Select Model", cohere_models)

st.sidebar.write(f"{model}: {cost[model]['description']}")
st.sidebar.write(f"{model} input token pricing: ${cost[model]['input_price_1M_tokens']} per million tokens")
st.sidebar.write(f"{model}: output token pricing: ${cost[model]['output_price_1M_tokens']} per million tokens")

#sidebar for temperature and max_tokens
st.sidebar.subheader("Model Parameters")

st.sidebar.title("Model Teperature")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.5, 0.1)

st.sidebar.title("Maximum Number of Tokens")
max_tokens = st.sidebar.slider("Max Tokens", 250, 4000, 2000, 10)

# Focus the LLM with a system prompt for all the models. Need a dropdown list that allows the user to select a system prompt such as, Code Assistant, Text Summariser, Idea Generator, etc.
st.sidebar.title("System Prompt")
system_prompts = ["Code Assistant", "Text Summariser", "Idea Generator", "Code Generator", 'Chat with Document']
system_prompt = st.sidebar.selectbox("Select System Prompt", system_prompts)

if system_prompt == "Code Assistant":
    pre_prompt = "You will act as an expert code assistant, here to help me develop clean and robust code. If code is provided, please make sure you return all code necessary to run the program."
elif system_prompt == "Text Summariser":
    pre_prompt = "You will act as a text summariser. Please provide me with the text to summarise it."
elif system_prompt == "Idea Generator":
    pre_prompt = "You will act as an idea generator. Please provide me with the topic to generate ideas."
elif system_prompt == "Code Generator":
    pre_prompt = "You will generate code based on the query. Please give me the code with no explanation"
elif system_prompt == "Chat with Document":
    pre_prompt = "The following information has been extracted from the document: \n"

# allow the user to upload a pdf, text, or csv file. Also allow the user to input a URL
st.sidebar.title("Document Loader")
st.sidebar.write("Upload a document or enter a URL to chat with the AI models.")
st.sidebar.write("Not available for Cohere")
document_types = ["None","PDF", "Text", "CSV", "Web"]
document_type = st.sidebar.selectbox("Select Document Type", document_types)

url=""
text_file=""
pdf_file=""
csv_file=""

if document_type == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
    # save file to path
    if pdf_file:
        with open('./upload_docs/file.pdf', 'wb') as f:
            f.write(pdf_file.read())
        document_loader = DocumentLoader()
        clean_text, length = document_loader.load_pdf('./upload_docs/file.pdf')

elif document_type == "Text":
    text_file = st.sidebar.file_uploader("Upload Text file", type=["txt"])
    if text_file:
        with open('./upload_docs/file.txt', 'w') as f:
            f.write(text_file.read())
        document_loader = DocumentLoader()
        clean_text, length = document_loader.load_text('./upload_docs/file.txt')

elif document_type == "CSV":
    csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if csv_file:
        with open('./upload_docs/file.csv', 'w') as f:
            f.write(csv_file.read())
        document_loader = DocumentLoader()
        clean_text, length = document_loader.load_csv('./upload_docs/file.csv')

elif document_type == "Web":
    url = st.sidebar.text_input("Enter URL")
    if url:
        document_loader = DocumentLoader()
        clean_text, length = document_loader.load_web(url)


#####################################OpenAI############################################


if ai_provider == 'OpenAI':
    if 'openai_client' not in st.session_state:
        st.session_state['openai_client'] = model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = ''

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        if url:
            pre_prompt = pre_prompt + clean_text
        elif text_file or csv_file or pdf_file:
            pre_prompt = pre_prompt + clean_text
        else:
            pre_prompt = pre_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            messages = [{"role": "system", "content": pre_prompt}] + st.session_state.messages
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = st.write_stream(stream)

            st.markdown(f"I was generated using: {model}")

        st.session_state.messages.append({"role": "assistant", "content": response})

        # calculate the cost of the chat
        cost, input_tokens, output_tokens = calculate_cost(prompt + pre_prompt, response, model)
        add_cost_to_session_state(cost)
        add_cost_to_total_cost(cost)
        st.write(f"Approximate cost of the chat: ${cost:.6f}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")


#####################################MistralAI############################################

elif ai_provider == 'MistralAI':
    if "mistral_model" not in st.session_state:
        st.session_state["mistral_model"] = model
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = ''

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    # Ensure system prompt is added as a dictionary if it doesn't exist
    if st.session_state["system_prompt"] and not any(
            message['role'] == "system" for message in st.session_state['messages']):
        system_message = {"role": "system", "content": st.session_state["system_prompt"]}
        st.session_state['messages'].insert(0, system_message)

    # Display all messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What is up?"):
        prompt = prompt
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        model_list = []

        # Generate response from MistralAI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if url:
                pre_prompt = pre_prompt + clean_text
            elif text_file or csv_file or pdf_file:
                pre_prompt = pre_prompt + clean_text
            else:
                pre_prompt = pre_prompt
            messages = [{"role": "system", "content": pre_prompt}] + st.session_state.messages
            for response in mistral_client.chat_stream(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
            ):
                for res in list(response):
                    model_list.append(res)

                model_used = model_list[1][1]

                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.markdown(f"I was generated using: {model_used}")

        # Append the assistant's response as a dictionary
        assistant_message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(assistant_message)
        cost, input_tokens, output_tokens = calculate_cost(pre_prompt + prompt, full_response, model)
        add_cost_to_session_state(cost)
        add_cost_to_total_cost(cost)
        st.write(
            f"Approximate cost of the chat: ${cost:.6f}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")


#####################################Anthropic############################################

elif ai_provider == 'Anthropic':
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = ''

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What is up?"):
        prompt = prompt
        if url:
            pre_prompt = pre_prompt + clean_text
        elif text_file or csv_file or pdf_file:
            pre_prompt = pre_prompt + clean_text
        else:
            pre_prompt = pre_prompt
        if not st.session_state.messages or st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.append({"role": "user", "content": pre_prompt + prompt})
            st.markdown(prompt)
            with st.chat_message("user"):
                message_placeholder = st.empty()
                full_response = ""

            # Generate and append assistant's response
            with anthropic_client.messages.stream(
                    max_tokens=max_tokens,
                    messages=st.session_state.messages,  # Include the entire conversation history
                    model=model,
                    temperature=temperature
            ) as stream:
                for text in stream.text_stream:
                    full_response += (text or "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                st.markdown(f"I was generated using: {model}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            cost, input_tokens, output_tokens = calculate_cost(pre_prompt + prompt, full_response, model)
            add_cost_to_session_state(cost)
            add_cost_to_total_cost(cost)
            st.write(
                f"Approximate cost of the chat: ${cost:.6f}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")


#####################################Cohere############################################

elif ai_provider == 'Cohere':
    if 'cohere_client' not in st.session_state:
        st.session_state['cohere_client'] = model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = ''

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        prompt = prompt
        if url:
            pre_prompt = pre_prompt + clean_text
        elif text_file or csv_file or pdf_file:
            pre_prompt = pre_prompt + clean_text
        else:
            pre_prompt = pre_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            messages = [{"role": "system", "content": pre_prompt}] + st.session_state.messages
            stream = cohere_client.chat_stream(
                model=model,
                message=str(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                connectors=[{"id": "web-search"}]
            )

            for event in stream:
                if event.event_type == "text-generation":
                    full_response += (event.text or "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.markdown(f"I was generated using: {model}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # calculate the cost of the chat
        cost, input_tokens, output_tokens = calculate_cost(prompt + pre_prompt, full_response, model)
        add_cost_to_session_state(cost)
        add_cost_to_total_cost(cost)
        st.write(f"Approximate cost of the chat: ${cost:.6f}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")


st.sidebar.markdown("---")
st.sidebar.title("Session State")
st.sidebar.subheader("Save and Load State")

col_1, col_2 = st.sidebar.columns(2)

with col_1:
    filename_save = st.text_input("Enter filename to save session state", "")
    if st.button("Save State"):
        if filename_save:
            save_state(filename_save)
            st.success(f"State saved to {filename_save}")
        else:
            st.error("Please enter a filename.")

with col_2:
    filename_load = st.text_input("Enter filename to load session state", "")
    if st.button("Load State"):
        try:
            load_state(filename_load)
            st.success(f"State loaded from {filename_load}")
            display_state_as_markdown()
        except FileNotFoundError:
            st.error("File not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

#list previous chat files in the previous_chats folder
previous_chats = os.listdir("./previous_chats")

if previous_chats:
    st.sidebar.title("Previous Chats")
    for chat in previous_chats:
        st.sidebar.write(chat)

#print the current total cost
if "total_cost" in st.session_state:
    st.sidebar.title("Total Cost of Session")
    st.sidebar.write(f"Total cost: ${st.session_state['total_cost']:.6f}")

st.sidebar.subheader("All Time Running Cost")
with open("total_cost.txt", "r") as f:
    total_cost = f.read()
    st.sidebar.write(f"Total cost: ${total_cost}")