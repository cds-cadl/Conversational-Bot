
import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import os
import shutil
import time
import pandas as pd
import nest_asyncio
import re
from datetime import datetime
import pytz

# Apply nest_asyncio to handle asynchronous operations in Streamlit
nest_asyncio.apply()

# ------------------------ Configuration ------------------------

# Set Streamlit page configuration
st.set_page_config(page_title="Conversational Bot", layout="wide")

# ------------------------ Secrets Management ------------------------

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------ Session State Initialization ------------------------

session_keys = [
    'conversation_history', 'log_data', 'rag_initialized',
    'generated_responses', 'system_prompt', 'additional_prompt',
    'system_query', 'num_responses', 'response_types', 'username', 'metric',
    'combined_system_prompt', 'user_query'
]
for key in session_keys:
    if key not in st.session_state:
        if key in ['conversation_history', 'log_data', 'generated_responses']:
            st.session_state[key] = []
        elif key == 'num_responses':
            st.session_state[key] = 2  # Default to 2 responses
        elif key == 'response_types':
            st.session_state[key] = "positive and negative"  # Default types
        elif key == 'system_prompt':
            st.session_state[key] = (
                "As Todd, respond to the following question in a conversational manner, "
                "keeping each response under 15 words for brevity and relevance. "
                "Focus on providing honest and personal answers that align with my perspective in the story. "
                "Provide responses labeled as 'Option 1:', 'Option 2:', etc."
            )
        elif key == 'additional_prompt':
            st.session_state[key] = ""
        elif key in ['username', 'metric', 'combined_system_prompt', 'user_query']:
            st.session_state[key] = ""
        else:
            st.session_state[key] = ""

# ------------------------ Initialize LightRAG ------------------------

@st.cache_resource
def initialize_lightrag():
    zip_path = "book_backup.zip"  #need to have the zip with DB in the dir
    extraction_path = "book_data/"  

    if not os.path.exists(extraction_path):
        shutil.unpack_archive(zip_path, extraction_path)
        st.write(f"📦 Book folder unzipped to: {extraction_path}")

    rag = LightRAG(
        working_dir=extraction_path,
        llm_model_func=gpt_4o_mini_complete
    )
    return rag

if not st.session_state.rag_initialized:
    with st.spinner("🔄 Initializing RAG system..."):
        rag = initialize_lightrag()
    st.session_state.rag = rag
    st.session_state.rag_initialized = True
else:
    rag = st.session_state.rag

# ------------------------ Helper Functions ------------------------

def generate_response(search_mode, system_query):
    try:
        start_time = time.time()
        response = rag.query(system_query, param=QueryParam(mode=search_mode))
        latency = time.time() - start_time
    except Exception as e:
        st.error(f"Error generating response: {e}")
        response = "Error generating response."
        latency = 0
    return response, latency

def update_conversation_history(speaker, text):
    st.session_state.conversation_history.append({'speaker': speaker, 'text': text})

def log_interaction(user_query, combined_system_prompt, system_query, search_mode, response, latency, comment=""):
    log_entry = {
        "Username": st.session_state.get('username', ''),
        "Metric": st.session_state.get('metric', ''),
        "User Query": user_query,
        "Combined System Prompt": combined_system_prompt,
        "System Query": system_query,
        "Search Mode": search_mode,
        "Response": response,
        "Latency (s)": f"{latency:.2f}",
        "Comment": comment
    }
    st.session_state.log_data.append(log_entry)

def parse_responses(raw_response, num_options):
    """
    Parses the raw response from the model using 'Option N:' labels.
    """
    responses = []
    for i in range(1, num_options + 1):
        pattern = rf'Option {i}: (.*?)(?=Option {i+1}:|$)'
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
            responses.append(f"Option {i}: {response_text}")
    return responses

def get_conversation_history(n=5):
    history = ''
    for entry in st.session_state.conversation_history[-n:]:
        speaker = entry['speaker']
        text = entry['text']
        history += f"{speaker}: {text}\n"
    return history

# ------------------------ User Interface ------------------------


st.title("🗣️ Conversational Bot Web App")

# ------------------------ Sidebar for User Information ------------------------

st.sidebar.header("User Information")
username = st.sidebar.text_input("Username (optional)", value=st.session_state.username)
metric = st.sidebar.text_input("Metric (optional)", value=st.session_state.metric)

st.session_state.username = username
st.session_state.metric = metric

# ------------------------ Main Input and Response Handling ------------------------


st.header("📝 Enter Your Queries")

input_container = st.container()
response_container = st.container()

with input_container:
    with st.form(key='query_form'):
        # Base System Prompt input (hidden by default)
        with st.expander("🛠️ **Base System Prompt** (Click to edit)"):
            system_prompt = st.text_area(
                "Edit the base system prompt:",
                value=st.session_state.system_prompt,
                height=150
            )

        # Additional System Prompt input (always visible)
        additional_prompt = st.text_area(
            "🛠️ **Additional System Prompt** (You can add more instructions here):",
            value=st.session_state.additional_prompt,
            height=100
        )

        # User Query input
        user_query = st.text_input("💬 **User Query**", "")

        # Number of Responses
        num_responses = st.number_input(
            "🔢 **Number of Responses**",
            min_value=1,
            max_value=5,
            value=st.session_state.num_responses,
            step=1
        )

        # Types of Responses
        response_types = st.text_input(
            "📝 **Types of Responses** (e.g., 'positive and negative', 'formal, informal, and humorous')",
            value=st.session_state.response_types
        )

        # Search Modes selection
        search_modes = ["naive", "local", "global", "hybrid"]
        selected_modes = st.multiselect(
            "🔍 **Select Search Modes**",
            options=search_modes,
            default=["naive"]  # Default to Naive
        )

        submit_button = st.form_submit_button(label='🔍 Submit')

    if submit_button and user_query.strip() != "":
        st.session_state.num_responses = num_responses
        st.session_state.response_types = response_types
        st.session_state.system_prompt = system_prompt
        st.session_state.additional_prompt = additional_prompt
        st.session_state.user_query = user_query

        update_conversation_history("🧑 User", user_query)

        conversation_history = get_conversation_history()

        combined_system_prompt = (
            f"{system_prompt}\n\n{additional_prompt}\n\n"
            f"Provide {num_responses} responses labeled as 'Option 1:', 'Option 2:', etc., "
            f"each reflecting {response_types} perspectives."
        )

        st.session_state.combined_system_prompt = combined_system_prompt

        system_query = f"{combined_system_prompt}\n\nConversation History:\n{conversation_history}\nUser Query: {user_query}"

        st.session_state.system_query = system_query

        with st.spinner("🔄 Generating responses..."):
            generated_responses = []
            for mode in selected_modes:
                response_text, latency = generate_response(mode, system_query)
                parsed_responses = parse_responses(response_text, num_responses)
                if not parsed_responses:
                    parsed_responses = [response_text.strip()]
                for resp in parsed_responses:
                    generated_responses.append({
                        'mode': mode,
                        'response': resp,
                        'latency': latency
                    })
            st.session_state.generated_responses = generated_responses

with response_container:
    if st.session_state.generated_responses:
        st.subheader("🔄 Generated Responses")
        response_options = []
        for idx, resp_info in enumerate(st.session_state.generated_responses, start=1):
            mode = resp_info['mode']
            response_text = resp_info['response']
            latency = resp_info['latency']
            option = f"Option {idx}: [{mode.capitalize()}] {response_text} (Latency: {latency:.2f}s)"
            response_options.append(option)

        for option in response_options:
            st.markdown(option)

        with st.form(key='selection_form'):
            selected_option = st.selectbox(
                "✅ **Choose Your Preferred Response:**",
                options=response_options
            )
            comment = st.text_area("📝 **Your Comment (optional):**", value="", height=100)
            confirm_button = st.form_submit_button(label='✅ Confirm Selection')

        if confirm_button and selected_option:
            selected_idx = response_options.index(selected_option)
            selected_response_info = st.session_state.generated_responses[selected_idx]
            chosen_mode = selected_response_info['mode']
            chosen_response_text = selected_response_info['response']
            chosen_latency = selected_response_info['latency']

            update_conversation_history("🧑🏻 Todd", chosen_response_text)

            log_interaction(
                user_query=st.session_state.user_query,
                combined_system_prompt=st.session_state.combined_system_prompt,
                system_query=st.session_state.system_query,
                search_mode=chosen_mode,
                response=chosen_response_text,
                latency=chosen_latency,
                comment=comment
            )

            st.session_state.generated_responses = []

            st.success("✅ Response selected and added to conversation history.")

# ------------------------ Sidebar for Downloading csv ------------------------

st.sidebar.header("📥 Download Logs")

def download_logs():
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        csv = df.to_csv(index=False)

        username = st.session_state.get('username', 'user') or 'user'
        metric = st.session_state.get('metric', 'metric') or 'metric'

        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        time_str = current_time.strftime("%Y%m%d_%H%M%S")

        file_name = f"{username}_{metric}_{time_str}.csv"

        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=file_name,
            mime='text/csv',
        )
    else:
        st.sidebar.write("🚫 No logs to download yet.")

# Call download_logs to render the updated state
download_logs()

# ------------------------ Conversation History ------------------------

history_container = st.container()

with history_container:
    st.header("💬 Conversation History")
    if st.session_state.conversation_history:
        st.markdown("""
            <style>
            .conversation-box {
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            </style>
            """, unsafe_allow_html=True)
        st.markdown('<div class="conversation-box">', unsafe_allow_html=True)
        for entry in st.session_state.conversation_history:
            speaker = entry['speaker']
            text = entry['text']
            st.markdown(f"**{speaker}:** {text}")
        st.markdown('</div>', unsafe_allow_html=True)
