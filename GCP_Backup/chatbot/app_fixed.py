import streamlit as st
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import os
import shutil
import time
import pandas as pd
import nest_asyncio
import re
import json
from datetime import datetime
import pytz
from git import Repo, Actor, GitCommandError
import tempfile
import asyncio

# Apply nest_asyncio to handle asynchronous operations
nest_asyncio.apply()

# ------------------------ Configuration ------------------------

st.set_page_config(page_title="Conversational Bot", layout="wide")

# ------------------------ Secrets Management ------------------------

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GIT_REPO_URL = st.secrets["GIT_REPO_URL"]
    GIT_DEPLOY_KEY = st.secrets["GIT_DEPLOY_KEY"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please ensure all required secrets are set.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------ Session State Initialization ------------------------

session_keys = [
    'conversation_history', 'log_data', 'rag_initialized',
    'generated_responses', 'system_prompt', 'additional_prompt',
    'system_query', 'num_responses', 'response_types', 'username', 'metric',
    'combined_system_prompt', 'user_query', 'chosen_response',
    'system_query_template', 'inference_mode'
]
for key in session_keys:
    if key not in st.session_state:
        if key in ['conversation_history', 'log_data', 'generated_responses']:
            st.session_state[key] = []
        elif key == 'rag_initialized':
            st.session_state[key] = False  # Initialize rag_initialized to False
        elif key == 'num_responses':
            st.session_state[key] = 2  # Default to 2 responses
        elif key == 'response_types':
            st.session_state[key] = "positive and negative"  # Default types
        elif key == 'system_prompt':
            st.session_state[key] = (
                "As Todd, respond to the following question in a conversational manner, "
                "keeping each response under 15 words for brevity and relevance. "
                "Focus on providing honest and personal answers that align with my perspective in the story."
            )
        elif key == 'additional_prompt':
            st.session_state[key] = ""
        elif key in ['username', 'metric', 'combined_system_prompt', 'user_query', 'chosen_response', 'system_query_template']:
            st.session_state[key] = ""
        elif key == 'inference_mode':
            st.session_state[key] = "Knowledge Base"  # Default mode

# ------------------------ Sidebar for Inference Mode Selection ------------------------

st.sidebar.header("Inference Mode")
inference_mode = st.sidebar.radio(
    "Choose your inference mode:",
    options=["Knowledge Base", "Direct Inference"],
    index=0,  # Default to Knowledge Base
    help="Select whether to use the book-based Knowledge Base or Direct Inference with the API."
)
st.session_state.inference_mode = inference_mode

# ------------------------ Initialize LightRAG ------------------------

@st.cache_resource
def initialize_lightrag():
    zip_path = "book_backup.zip"  # Ensure this file is in the project directory
    extraction_path = "book_data/"

    if not os.path.exists(extraction_path):
        shutil.unpack_archive(zip_path, extraction_path)
        st.write(f"📦 Book folder unzipped to: {extraction_path}")

    rag = LightRAG(
        working_dir=extraction_path,
        llm_model_func=gpt_4o_mini_complete
    )
    return rag

if st.session_state.inference_mode == "Knowledge Base":
    if not st.session_state.rag_initialized:
        with st.spinner("🔄 Initializing RAG system..."):
            rag = initialize_lightrag()
        st.session_state.rag = rag
        st.session_state.rag_initialized = True
    else:
        rag = st.session_state.rag

# ------------------------ Helper Functions ------------------------

def generate_response(system_query, kb_mode=None):
    try:
        start_time = time.time()
        if st.session_state.inference_mode == "Direct Inference":
            # Direct inference: bypass the KB and call the API directly
            result = gpt_4o_mini_complete(system_query)
            if asyncio.iscoroutine(result):
                response = asyncio.get_event_loop().run_until_complete(result)
            else:
                response = result
        else:
            # Knowledge Base mode: use the KB with the selected search mode.
            # Use kb_mode if provided; otherwise default to "naive"
            mode = kb_mode if kb_mode is not None else "naive"
            response = rag.query(system_query, param=QueryParam(mode=mode))
        latency = time.time() - start_time
    except Exception as e:
        st.error(f"Error generating response: {e}")
        response = "Error generating response."
        latency = 0
    return response, latency

def update_conversation_history(speaker, text):
    st.session_state.conversation_history.append({'speaker': speaker, 'text': text})

def log_interaction(user_query, combined_system_prompt, system_query, generated_responses, chosen_response):
    log_entry = {
        "Username": st.session_state.get('username', ''),
        "Metric": st.session_state.get('metric', ''),
        "User Query": user_query,
        "Combined System Prompt": combined_system_prompt,
        "System Query": system_query,
        "Generated Responses": json.dumps(generated_responses, ensure_ascii=False),
        "Chosen Response": json.dumps(chosen_response, ensure_ascii=False),
        "Inference Mode": st.session_state.inference_mode,
        "Timestamp": datetime.utcnow().isoformat() + "Z"
    }
    st.session_state.log_data.append(log_entry)

def parse_responses(raw_response):
    """
    Parses the raw JSON response from the model.
    Extracts JSON content even if there's additional text.
    """
    try:
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array found in the response.")
        json_str = json_match.group()
        responses = json.loads(json_str)
        if not isinstance(responses, list):
            raise ValueError("Parsed JSON is not a list.")
        parsed_responses = []
        for item in responses:
            option_number = item.get('option_number', 'N/A')
            response_text = item.get('response', '')
            if response_text:
                parsed_responses.append({
                    "Option Number": option_number,
                    "Response Text": response_text
                })
        return parsed_responses
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Failed to parse responses: {e}")
        st.write(f"Raw response: {raw_response}")
        return [{"Option Number": "N/A", "Response Text": raw_response.strip()}]

def get_conversation_history(n=5):
    history = ''
    for entry in st.session_state.conversation_history[-n:]:
        speaker = entry['speaker']
        text = entry['text']
        history += f"{speaker}: {text}\n"
    return history

def save_logs_to_git(selected_user, log_df):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Write deploy key to a temporary file
            deploy_key_fd, deploy_key_path = tempfile.mkstemp()
            with os.fdopen(deploy_key_fd, 'w') as key_file:
                key_file.write(GIT_DEPLOY_KEY)
            os.chmod(deploy_key_path, 0o600)
            git_ssh_command = f'ssh -i {deploy_key_path} -o StrictHostKeyChecking=no'
            repo = Repo.clone_from(
                GIT_REPO_URL,
                tmpdirname,
                branch='main',
                depth=1,
                env={'GIT_SSH_COMMAND': git_ssh_command}
            )
            # Create a "logs" folder, then a folder for the selected user within that folder
            logs_folder = os.path.join(tmpdirname, "KB_DIRECT")
            os.makedirs(logs_folder, exist_ok=True)
            user_folder = os.path.join(logs_folder, selected_user)
            os.makedirs(user_folder, exist_ok=True)
            log_file_path = os.path.join(user_folder, "logs.csv")
            if os.path.exists(log_file_path):
                existing_df = pd.read_csv(log_file_path)
                updated_df = pd.concat([existing_df, log_df], ignore_index=True)
            else:
                updated_df = log_df
            updated_df.to_csv(log_file_path, index=False)
            repo.index.add([os.path.relpath(log_file_path, tmpdirname)])
            author = Actor("Streamlit App", "app@example.com")
            commit_message = f"Update logs for {selected_user} at {datetime.utcnow().isoformat()}Z"
            repo.index.commit(commit_message, author=author, committer=author)
            origin = repo.remote(name='origin')
            origin.push()
            os.remove(deploy_key_path)
    except GitCommandError as git_err:
        st.error(f"Git command error: {git_err}")
    except Exception as e:
        st.error(f"An error occurred while saving logs to Git: {e}")

# ------------------------ User Interface ------------------------

st.title("🗣️ Conversational Bot Web App")

# ------------------------ Sidebar for User Information and Git Integration ------------------------

st.markdown(
    """
    <style>
    .sidebar .sidebar-content h1 {
        font-size: 24px; 
        color: #ff6347; 
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("CADL🫶🏻")
st.sidebar.header("User Information")

predefined_users = ["Jeff", "Pam", "Antara", "Jenna", "Shalini", "Manohar", "Kat", "Mike"]

selected_user = st.sidebar.selectbox(
    "Select Your Name",
    options=predefined_users,
    index=0
)

metric = st.sidebar.text_input(
    "Metric (optional)",
    value=st.session_state.metric,
    help="Enter an optional metric for logging purposes.",
    placeholder="e.g., Steerability"
)

st.session_state.username = selected_user
st.session_state.metric = metric



# ------------------------ Main Input and Response Handling ------------------------

st.header("📝 Enter Your Queries")

input_container = st.container()
response_container = st.container()

with input_container:
    with st.form(key='query_form'):
        st.markdown("### 📄 Configure Your Query")
        
        with st.expander("🛠️ **Base System Prompt** (Click to edit)"):
            system_prompt = st.text_area(
                "Edit the base system prompt:",
                value=st.session_state.system_prompt,
                height=150,
                help="Modify the base instructions for the conversational agent.",
                placeholder="As Todd, respond to the following question..."
            )

        additional_prompt = st.text_area(
            "🛠️ **Additional System Prompt** (You can add more instructions here):",
            value=st.session_state.additional_prompt,
            height=100,
            help="Add supplementary instructions or context for the conversational agent.",
            placeholder="You can add more instructions here..."
        )

        st.markdown("---")

        user_query = st.text_input(
            "💬 **User Query**",
            "",
            help="Enter your query here. For example: 'How was your weekend?'",
            placeholder="Type your question..."
        )

        num_responses = st.number_input(
            "🔢 **Number of Responses**",
            min_value=1,
            max_value=5,
            value=st.session_state.num_responses,
            step=1,
            help="Select the number of responses you would like to receive."
        )

        response_types = st.text_input(
            "📝 **Types of Responses**",
            value=st.session_state.response_types,
            help="Specify the types of responses. For example: 'positive and negative', 'formal, informal, and humorous'.",
            placeholder="e.g., positive and negative"
        )

        # Only show KB search modes if using Knowledge Base inference
        if st.session_state.inference_mode == "Knowledge Base":
            selected_search_modes = st.multiselect(
                "🔍 **Select KB Search Modes**",
                options=["naive", "local", "global", "hybrid"],
                default=["naive"],
                help="Choose the search modes to use for generating responses."
            )
        else:
            selected_search_modes = None

        st.markdown("---")

        generate_template = st.form_submit_button(label='🛠️ Generate Template')

        if generate_template:
            if not user_query.strip():
                st.warning("💡 Please enter a user query to generate the system query template.")
            else:
                combined_system_prompt = (
                    f"{system_prompt}\n\n{additional_prompt}\n\n"
                    f"Provide {num_responses} responses in JSON format as a list of objects, "
                    f"each with 'option_number' and 'response' fields, reflecting {response_types} perspectives. "
                    f"**Return only the JSON without any additional text.**"
                )
                conversation_history = get_conversation_history()
                system_query = f"{combined_system_prompt}\n\nConversation History:\n{conversation_history}\nUser Query: {user_query}"
                st.session_state.system_query_template = system_query
                with st.expander("📄 **System Query Template**"):
                    st.code(system_query, language='plaintext')

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
            f"Provide {num_responses} responses in JSON format as a list of objects, "
            f"each with 'option_number' and 'response' fields, reflecting {response_types} perspectives. "
            f"**Return only the JSON without any additional text.**"
        )
        st.session_state.combined_system_prompt = combined_system_prompt

        system_query = f"{combined_system_prompt}\n\nConversation History:\n{conversation_history}\nUser Query: {user_query}"
        st.session_state.system_query = system_query

        with st.spinner("🔄 Generating responses..."):
            generated_responses = []
            # For Direct Inference, just call generate_response once.
            if st.session_state.inference_mode == "Direct Inference":
                response_text, latency = generate_response(system_query)
                parsed_responses = parse_responses(response_text)
                if not parsed_responses:
                    parsed_responses = [{"Option Number": "N/A", "Response Text": response_text.strip()}]
                for resp in parsed_responses:
                    generated_responses.append({
                        "Mode": st.session_state.inference_mode,
                        "Response": resp["Response Text"],
                        "Latency (s)": round(latency, 2)
                    })
            else:
                # For KB mode, iterate over each selected search mode.
                if not selected_search_modes:
                    selected_search_modes = ["naive"]
                for mode in selected_search_modes:
                    response_text, latency = generate_response(system_query, kb_mode=mode)
                    parsed_responses = parse_responses(response_text)
                    if not parsed_responses:
                        parsed_responses = [{"Option Number": "N/A", "Response Text": response_text.strip()}]
                    for resp in parsed_responses:
                        generated_responses.append({
                            "Mode": f"{st.session_state.inference_mode} ({mode})",
                            "Response": resp["Response Text"],
                            "Latency (s)": round(latency, 2)
                        })
            st.session_state.generated_responses = generated_responses

with response_container:
    if st.session_state.generated_responses:
        st.subheader("🔄 Generated Responses")
        response_options = []
        for idx, resp_info in enumerate(st.session_state.generated_responses, start=1):
            mode = resp_info['Mode']
            response_text = resp_info['Response']
            latency = resp_info['Latency (s)']
            option = f"{response_text} (Mode: {mode}, Latency: {latency:.2f}s)"
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
            chosen_mode = selected_response_info['Mode']
            chosen_response_text = selected_response_info['Response']
            chosen_latency = selected_response_info['Latency (s)']
            update_conversation_history("🧑🏻 Todd", chosen_response_text)
            generated_responses_structured = st.session_state.generated_responses
            log_interaction(
                user_query=st.session_state.user_query,
                combined_system_prompt=st.session_state.combined_system_prompt,
                system_query=st.session_state.system_query,
                generated_responses=generated_responses_structured,
                chosen_response={
                    "Mode": chosen_mode,
                    "Response": chosen_response_text,
                    "Latency (s)": chosen_latency,
                    "Comment": comment
                }
            )
            st.session_state.generated_responses = []
            st.success("✅ Response selected and added to conversation history.")
            if st.session_state.log_data:
                log_df = pd.DataFrame(st.session_state.log_data)
                save_logs_to_git(selected_user, log_df)
                st.session_state.log_data = []

# ------------------------ Sidebar for Downloading CSV ------------------------

st.sidebar.header("📥 Download Logs")
def download_logs():
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        df['Generated Responses'] = df['Generated Responses'].apply(lambda x: json.dumps(json.loads(x), ensure_ascii=False, indent=2))
        df['Chosen Response'] = df['Chosen Response'].apply(lambda x: json.dumps(x, ensure_ascii=False, indent=2))
        csv = df.to_csv(index=False)
        selected_user = st.session_state.get('username', 'user') or 'user'
        metric = st.session_state.get('metric', 'metric') or 'metric'
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{selected_user}_{metric}_{time_str}.csv"
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=file_name,
            mime='text/csv',
        )
    else:
        st.sidebar.write("🚫 No logs to download yet.")
download_logs()

# ------------------------ Conversation History ------------------------

history_container = st.container()
with history_container:
    st.header("💬 Conversation History")
    if st.session_state.conversation_history:
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="conversation-box">', unsafe_allow_html=True)
        for entry in st.session_state.conversation_history:
            speaker = entry['speaker']
            text = entry['text']
            st.markdown(f"**{speaker}:** {text}")
        st.markdown('</div>', unsafe_allow_html=True)

