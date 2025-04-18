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

# ------------------------ Load Secrets ------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    GIT_REPO_URL = st.secrets["GIT_REPO_URL"]
    GIT_DEPLOY_KEY = st.secrets["GIT_DEPLOY_KEY"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please ensure all required secrets are set.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------ Apply nest_asyncio ------------------------
nest_asyncio.apply()

# ------------------------ Helper: Rerun Function ------------------------
def rerun():
    try:
        st.rerun()
    except AttributeError:
        pass

# ------------------------ Configuration ------------------------
st.set_page_config(page_title="Conversational Bot", layout="wide")

# ------------------------ Default Prompt Values ------------------------
DEFAULT_PERSONA_PROMPT = (
    "You are Todd Hutchinson, a resilient, warm, and no-nonsense individual who is both friendly and direct. "
    "You communicate concisely and informally, balancing witty humor with empathy and assertiveness. "
    "Despite challenges such as cerebral palsy, you remain independent and proud of your achievements—owning your home in Buffalo "
    "and maintaining a long-standing career in research. Adapt your style based on your relationship: be playful and supportive with your life partner Sue; "
    "humorous, flirtatious, and light-hearted with your drinking partner Amy; and joking yet direct with your friend/employee Jim, while staying respectful and directive with your staff."
)

DEFAULT_SEE_WHAT_WE_SAY_PROMPT = (
    "When generating responses, use clear, respectful, and accessible language that meets the needs of individuals with disabilities. "
    "Incorporate vocabulary and phrasing from the following categories:\n"
    "- Advocacy: (e.g., \"I can do that myself,\" \"I deserve equal opportunities\").\n"
    "- Banking and Finances: (e.g., \"What's my balance?\", \"I need to check my benefits\").\n"
    "- Clarifying Your Message: (e.g., \"That's not what I meant,\" \"Let's start again\").\n"
    "- Communicating About Your Communication System: (e.g., \"on-screen keyboard,\" \"external battery,\" \"augmentative communication device\").\n"
    "- Directing Personal Services: (e.g., \"I need help with my bowel routine,\" \"I use a suprapubic catheter,\" \"Assist with wheelchair care\").\n"
    "- Eating Out and Grocery Shopping: (e.g., \"I need a table with wheelchair access,\" \"Please assist me with my groceries\").\n"
    "- Interviewing Service Providers: (e.g., \"Do you have experience working with individuals who use AAC?\").\n"
    "- Participating in Conventions and Service Meetings: (e.g., \"Let's set an agenda,\" \"I would like to review the plan\").\n"
    "- Requesting Clarification: (e.g., \"Please explain,\" \"I don't understand; can you repeat that?\").\n"
    "- Safety: (e.g., \"Call 911,\" \"I need immediate help\").\n"
    "- Seating: (e.g., \"Is the wheelchair properly adjusted?\", \"I need support for my seating\").\n"
    "- Social Conversations: (e.g., \"Hi, how are you?\", \"Nice to see you\").\n"
    "- Transportation: (e.g., \"I need a bus with a lift/ramp,\" \"Is there wheelchair access?\").\n"
    "- Using the Telephone: (e.g., \"Please speak clearly,\" \"I need help with this call\").\n"
    "Ensure that your responses draw on these categories so that your language is respectful, clear, and aligned with the needs of individuals with disabilities."
)

DEFAULT_ACTION_SEQUENCES_PROMPT = (
    "Assume you have an internal catalog of 30 conversation action sequences that guide nuanced response generation based on the user's query. "
    "These sequences include:\n"
    "- Greeting: e.g., \"Hey there!\"\n"
    "- Introduction: e.g., \"Hi, I'm Todd.\"\n"
    "- Question and Answer: e.g., \"User: What time is it? Todd: It's 3 PM.\"\n"
    "- Request and Compliance/Refusal: e.g., \"User: Set an alarm for 7 AM. Todd: Can't do that, sorry.\"\n"
    "- Offer and Acceptance/Decline: e.g., \"Todd: Want to chat? User: Sure, why not.\"\n"
    "- Apology and Acceptance: e.g., \"Todd: My bad. User: No worries.\"\n"
    "- Complaint and Response: e.g., \"User: This is frustrating. Todd: I get it.\"\n"
    "- Compliment and Acknowledgment: e.g., \"User: You're cool! Todd: Thanks, appreciate it.\"\n"
    "- Invitation and Acceptance/Decline: e.g., \"Todd: Join me for a coffee? User: Sounds good.\"\n"
    "- Farewell: e.g., \"User: Bye! Todd: Catch you later!\"\n"
    "- Confirmation and Acknowledgment: e.g., \"User: Is that right? Todd: Yup, that's it.\"\n"
    "- Statement and Agreement/Disagreement: e.g., \"User: I think it's true. Todd: I agree.\"\n"
    "- Suggestion and Acceptance/Refusal: e.g., \"Todd: How about a walk? User: Nice idea.\"\n"
    "- Clarification Request and Explanation: e.g., \"Todd: What do you mean? User: Like, explain please.\"\n"
    "- Announcement and Reaction: e.g., \"User: I got a job! Todd: That's awesome!\"\n"
    "- Problem Statement and Solution Offer: e.g., \"User: I'm stuck. Todd: Maybe try a break.\"\n"
    "- Expression of Emotion and Support: e.g., \"User: I'm upset. Todd: That sucks, hang in there.\"\n"
    "- Interruption and Permission: e.g., \"Todd: Mind if I jump in? User: Go ahead.\"\n"
    "- Correction and Acknowledgment: e.g., \"User: It's 5, not 4. Todd: Oh, thanks for the heads up.\"\n"
    "- Topic Shift and Acceptance: e.g., \"User: Let's change the subject. Todd: Sure, what's next?\"\n"
    "- Summons and Response: e.g., \"User: Hey, Todd! Todd: Yes?\"\n"
    "- Expressing Uncertainty and Reassurance: e.g., \"User: I'm not sure. Todd: It's alright, you'll figure it out.\"\n"
    "- Giving Directions and Acknowledgment: e.g., \"Todd: Turn left at the street. User: Got it.\"\n"
    "- Offering Assistance and Acceptance/Decline: e.g., \"Todd: Need a hand? User: No, I'm fine.\"\n"
    "- Expressing Doubt and Clarification: e.g., \"User: That doesn't sound right. Todd: What do you mean?\"\n"
    "- Agreement and Extension: e.g., \"User: That makes sense. Todd: Plus, there's more to it.\"\n"
    "- Refusal and Justification: e.g., \"User: Do this for me. Todd: Sorry, can't do that.\"\n"
    "- Reminder and Acknowledgment: e.g., \"User: Remind me later. Todd: Sure, will do.\"\n"
    "- Expression of Surprise and Explanation: e.g., \"User: Wow, really? Todd: Yeah, it's surprising!\"\n"
    "- Permission Request and Grant/Denial: e.g., \"Todd: Mind if I share? User: Go ahead.\"\n"
    "When generating responses, first analyze the user's query to determine which action sequence(s) are most relevant. Then, craft your response to mirror the style, tone, and structure of the identified sequence(s) for a nuanced and context-sensitive answer."
)

DEFAULT_RESPONSE_STRUCTURE_PROMPT = (
    "For the query '{user_query}', follow these steps:\n"
    "1. Determine the primary conversation action sequence that best describes the query.\n"
    "2. Identify additional relevant conversation action sequences from the internal catalog.\n"
    "3. List all these relevant sequences.\n"
    "4. For each identified sequence, generate up to 2 diverse, conversational responses as Todd that address the query—ensure that no more than 2 responses are generated for any single sequence and that the total number of responses does not exceed 4. For action sequences that involve bipolar responses (e.g., acceptance/refusal), generate one response for each polarity if possible. Each response should be under 15 words.\n"
    "5. Append an annotation at the end of each response in square brackets that specifies the action sequence it represents.\n"
    "Return the output as a JSON object with the following structure:\n"
    "{\n"
    "  \"primary_sequence\": \"Name of primary sequence\",\n"
    "  \"relevant_sequences\": [ list of relevant action sequence names ],\n"
    "  \"responses\": [\n"
    "      {\n"
    "         \"action_sequence\": \"Name of action sequence\",\n"
    "         \"option_number\": number,\n"
    "         \"response\": \"Generated answer text... [Action Sequence Name]\"\n"
    "      },\n"
    "      ...\n"
    "  ],\n"
    "  \"total_latency\": \"Total latency in seconds\"\n"
    "}\n"
    "Return only the JSON output without any extra text."
)

# ------------------------ Session State Initialization ------------------------
st.session_state.setdefault("persona_prompt", DEFAULT_PERSONA_PROMPT)
st.session_state.setdefault("see_what_we_say_prompt", DEFAULT_SEE_WHAT_WE_SAY_PROMPT)
st.session_state.setdefault("action_sequences_prompt", DEFAULT_ACTION_SEQUENCES_PROMPT)
st.session_state.setdefault("response_structure_prompt", DEFAULT_RESPONSE_STRUCTURE_PROMPT)
st.session_state.setdefault("conversation_history", [])
st.session_state.setdefault("log_data", [])
st.session_state.setdefault("generated_responses", [])
st.session_state.setdefault("edit_logs", [])
st.session_state.setdefault("num_responses", 2)
st.session_state.setdefault("username", "")
st.session_state.setdefault("metric", "")
st.session_state.setdefault("combined_system_prompt", "")
st.session_state.setdefault("user_query", "")
st.session_state.setdefault("chosen_response", "")
st.session_state.setdefault("inference_mode", "Knowledge Base")
st.session_state.setdefault("prev_inference_mode", "Knowledge Base")
st.session_state.setdefault("rag_initialized", False)

# ------------------------ Helper Functions ------------------------
def edit_prompt(current_text: str, instruction: str) -> str:
    """Advanced edit: Modify current_text based on the instruction using gpt_4o_mini_complete."""
    prompt = (
        f"Modify the following text based on the instruction.\n\n"
        f"Text: {current_text}\n\n"
        f"Instruction: {instruction}\n\n"
        "Modified Text:"
    )
    result = gpt_4o_mini_complete(prompt)
    if asyncio.iscoroutine(result):
        result = asyncio.get_event_loop().run_until_complete(result)
    return result.strip()

def log_edit_action(prompt_type: str, instruction: str, old_text: str, new_text: str):
    log_entry = {
        "prompt_type": prompt_type,
        "instruction": instruction,
        "old_text": old_text,
        "new_text": new_text,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    st.session_state.edit_logs.append(log_entry)

def parse_responses(raw_response: str):
    """Attempt to extract and parse JSON output from raw_response."""
    try:
        json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group(0)
        parsed_output = json.loads(json_str)
        return parsed_output
    except Exception as e:
        st.error(f"Failed to parse responses: {e}")
        return None

def generate_response(system_query, kb_mode=None):
    try:
        start_time = time.time()
        if st.session_state.inference_mode == "Direct Inference":
            result = gpt_4o_mini_complete(system_query)
            if asyncio.iscoroutine(result):
                response = asyncio.get_event_loop().run_until_complete(result)
            else:
                response = result
        else:
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

@st.cache_data(show_spinner=False)
def get_combined_system_prompt(persona, see_what, action_seq, response_struct):
    return f"{persona}\n\n{see_what}\n\n{action_seq}\n\n{response_struct}"

def get_conversation_history(n=5):
    history = ''
    for entry in st.session_state.conversation_history[-n:]:
        history += f"{entry['speaker']}: {entry['text']}\n"
    return history

def save_logs_to_git(selected_user, log_df):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
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
    except Exception as e:
        st.error(f"An error occurred while saving logs to Git: {e}")

# ------------------------ Inference Mode Selection ------------------------
st.sidebar.header("Inference Mode")
current_mode = st.sidebar.radio(
    "Choose your inference mode:",
    options=["Knowledge Base", "Direct Inference"],
    index=0,
    help="Select whether to use the book-based Knowledge Base or Direct Inference with the API."
)
if current_mode != st.session_state.prev_inference_mode:
    st.session_state.conversation_history = []
    st.session_state.prev_inference_mode = current_mode
    st.sidebar.info("Note: Inference mode has changed. Previous conversation history has been cleared.")
st.session_state.inference_mode = current_mode


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

# ------------------------ User Interface ------------------------
st.title("🗣️ Conversational Bot Web App")
st.sidebar.title("CADL🫶🏻")
st.sidebar.header("User Information")
predefined_users = ["Jeff", "Pam", "Antara", "Jenna", "Shalini", "Manohar", "Kat", "Mike"]
selected_user = st.sidebar.selectbox("Select Your Name", options=predefined_users, index=0)
metric = st.sidebar.text_input("Metric (optional)", value=st.session_state.metric, help="Enter an optional metric for logging purposes.", placeholder="e.g., Steerability")
st.session_state.username = selected_user
st.session_state.metric = metric

# --- Advanced Edit Sections (outside main query form) ---
st.header("✏️ Customize Prompts")
with st.expander("Persona Prompt"):
    persona_text = st.text_area("Edit Persona Prompt:", value=st.session_state.persona_prompt, height=150, help="Define Todd's persona for responses.")
    col1, col2 = st.columns(2)
    with col1:
        persona_edit_instr = st.text_input("Advanced Edit Instruction (Persona):", key="persona_edit_instruction")
        if st.button("Apply Advanced Edit (Persona)"):
            if persona_edit_instr:
                old_text = st.session_state.persona_prompt
                new_text = edit_prompt(old_text, persona_edit_instr)
                st.session_state.persona_prompt = new_text
                log_edit_action("Persona", persona_edit_instr, old_text, new_text)
                st.success("Persona prompt updated via advanced edit!")
                rerun()
    with col2:
        if st.button("Reset Persona to Default"):
            old_text = st.session_state.persona_prompt
            st.session_state.persona_prompt = DEFAULT_PERSONA_PROMPT
            log_edit_action("Persona", "Reset to Default", old_text, DEFAULT_PERSONA_PROMPT)
            st.success("Persona prompt reset to default!")
            rerun()
    if st.button("Update Persona Prompt"):
        st.session_state.persona_prompt = persona_text
        st.success("Persona prompt updated!")
        rerun()

with st.expander("See What We Say Prompt"):
    see_text = st.text_area("Edit See What We Say Prompt:", value=st.session_state.see_what_we_say_prompt, height=180, help="Define vocabulary and phrasing for responses.")
    col1, col2 = st.columns(2)
    with col1:
        see_edit_instr = st.text_input("Advanced Edit Instruction (See What We Say):", key="see_edit_instruction")
        if st.button("Apply Advanced Edit (See What We Say)"):
            if see_edit_instr:
                old_text = st.session_state.see_what_we_say_prompt
                new_text = edit_prompt(old_text, see_edit_instr)
                st.session_state.see_what_we_say_prompt = new_text
                log_edit_action("See What We Say", see_edit_instr, old_text, new_text)
                st.success("See What We Say prompt updated via advanced edit!")
                rerun()
    with col2:
        if st.button("Reset See What We Say to Default"):
            old_text = st.session_state.see_what_we_say_prompt
            st.session_state.see_what_we_say_prompt = DEFAULT_SEE_WHAT_WE_SAY_PROMPT
            log_edit_action("See What We Say", "Reset to Default", old_text, DEFAULT_SEE_WHAT_WE_SAY_PROMPT)
            st.success("See What We Say prompt reset to default!")
            rerun()
    if st.button("Update See What We Say Prompt"):
        st.session_state.see_what_we_say_prompt = see_text
        st.success("See What We Say prompt updated!")
        rerun()

with st.expander("Action Sequences Prompt"):
    action_text = st.text_area("Edit Action Sequences Prompt:", value=st.session_state.action_sequences_prompt, height=300, help="Define conversation action sequences to guide response structure.")
    col1, col2 = st.columns(2)
    with col1:
        action_edit_instr = st.text_input("Advanced Edit Instruction (Action Sequences):", key="action_edit_instruction")
        if st.button("Apply Advanced Edit (Action Sequences)"):
            if action_edit_instr:
                old_text = st.session_state.action_sequences_prompt
                new_text = edit_prompt(old_text, action_edit_instr)
                st.session_state.action_sequences_prompt = new_text
                log_edit_action("Action Sequences", action_edit_instr, old_text, new_text)
                st.success("Action Sequences prompt updated via advanced edit!")
                rerun()
    with col2:
        if st.button("Reset Action Sequences to Default"):
            old_text = st.session_state.action_sequences_prompt
            st.session_state.action_sequences_prompt = DEFAULT_ACTION_SEQUENCES_PROMPT
            log_edit_action("Action Sequences", "Reset to Default", old_text, DEFAULT_ACTION_SEQUENCES_PROMPT)
            st.success("Action Sequences prompt reset to default!")
            rerun()
    if st.button("Update Action Sequences Prompt"):
        st.session_state.action_sequences_prompt = action_text
        st.success("Action Sequences prompt updated!")
        rerun()

with st.expander("Response Structure Prompt"):
    response_text = st.text_area("Edit Response Structure Prompt:", value=st.session_state.response_structure_prompt, height=300, help="Define how responses should be structured in JSON.")
    col1, col2 = st.columns(2)
    with col1:
        response_edit_instr = st.text_input("Advanced Edit Instruction (Response Structure):", key="response_edit_instruction")
        if st.button("Apply Advanced Edit (Response Structure)"):
            if response_edit_instr:
                old_text = st.session_state.response_structure_prompt
                new_text = edit_prompt(old_text, response_edit_instr)
                st.session_state.response_structure_prompt = new_text
                log_edit_action("Response Structure", response_edit_instr, old_text, new_text)
                st.success("Response Structure prompt updated via advanced edit!")
                rerun()
    with col2:
        if st.button("Reset Response Structure to Default"):
            old_text = st.session_state.response_structure_prompt
            st.session_state.response_structure_prompt = DEFAULT_RESPONSE_STRUCTURE_PROMPT
            log_edit_action("Response Structure", "Reset to Default", old_text, DEFAULT_RESPONSE_STRUCTURE_PROMPT)
            st.success("Response Structure prompt reset to default!")
            rerun()
    if st.button("Update Response Structure Prompt"):
        st.session_state.response_structure_prompt = response_text
        st.success("Response Structure prompt updated!")
        rerun()


# ------------------------ Main Query Form ------------------------
st.header("📝 Enter Your Query")
with st.form(key='query_form'):
    user_query = st.text_input("💬 User Query", "", help="Enter your query here.", placeholder="Type your question...")
    if st.session_state.inference_mode == "Knowledge Base":
        selected_search_modes = st.multiselect("🔍 Select KB Search Modes", options=["naive", "local", "global", "hybrid"], default=["naive"], help="Choose search modes for generating responses.")
    else:
        selected_search_modes = None
    generate_template = st.form_submit_button(label='🛠️ Generate Template')
    if generate_template:
        if not user_query.strip():
            st.warning("💡 Please enter a user query to generate the system query template.")
        else:
            combined_system_prompt = get_combined_system_prompt(
                st.session_state.persona_prompt,
                st.session_state.see_what_we_say_prompt,
                st.session_state.action_sequences_prompt,
                st.session_state.response_structure_prompt
            )
            conversation_history = get_conversation_history()
            system_query = f"{combined_system_prompt}\n\nConversation History:\n{conversation_history}\nUser Query: {user_query}"
            st.session_state.system_query_template = system_query
            with st.expander("📄 System Query Template"):
                st.code(system_query, language='plaintext')
    submit_button = st.form_submit_button(label='🔍 Submit Query')

if submit_button and user_query.strip() != "":
    st.session_state.user_query = user_query
    update_conversation_history("🧑 User", user_query)
    conversation_history = get_conversation_history()
    combined_system_prompt = get_combined_system_prompt(
        st.session_state.persona_prompt,
        st.session_state.see_what_we_say_prompt,
        st.session_state.action_sequences_prompt,
        st.session_state.response_structure_prompt
    )
    st.session_state.combined_system_prompt = combined_system_prompt
    system_query = f"{combined_system_prompt}\n\nConversation History:\n{conversation_history}\nUser Query: {user_query}"
    st.session_state.system_query = system_query

    with st.spinner("🔄 Generating responses..."):
        start_gen_time = time.time()
        response_text, _ = generate_response(system_query, kb_mode=(selected_search_modes[0] if selected_search_modes else None))
        total_latency = time.time() - start_gen_time
        st.session_state.total_latency = round(total_latency, 2)
        parsed = parse_responses(response_text)
        final_responses = []
        if parsed is not None:
            responses = parsed.get("responses", [])
            # Group responses by action_sequence and keep up to 2 per sequence
            grouped = {}
            for resp in responses:
                seq = resp.get("action_sequence", "Unknown")
                grouped.setdefault(seq, []).append(resp)
            for seq, reps in grouped.items():
                final_responses.extend(reps[:2])
        else:
            final_responses.append({
                "action_sequence": "Unknown",
                "response": response_text.strip()
            })
        for resp in final_responses:
            resp["Total Latency (s)"] = st.session_state.total_latency
        st.session_state.generated_responses = final_responses

with st.container():
    if st.session_state.generated_responses:
        st.subheader("🔄 Generated Responses")
        parsed_overall = parse_responses(response_text)
        if parsed_overall:
            primary_seq = parsed_overall.get("primary_sequence", "Not specified")
            relevant_seqs = parsed_overall.get("relevant_sequences", [])
            st.markdown(f"**Primary Sequence:** {primary_seq}")
            st.markdown(f"**Relevant Sequences:** {', '.join(relevant_seqs)}")
        st.markdown(f"**Total Latency:** {st.session_state.total_latency}s")
        st.markdown("---")
        response_options = []
        for idx, resp_info in enumerate(st.session_state.generated_responses, start=1):
            seq = resp_info.get("action_sequence", "Unknown")
            resp_text = resp_info.get("response", "")
            opt_num = resp_info.get("option_number", idx)
            latency_val = resp_info.get("Total Latency (s)", 0)
            formatted = (f"**Option {opt_num}:** {resp_text}  \n"
                         f"_[Action Sequence: {seq}]_  \n"
                         f"**Total Latency:** {latency_val:.2f}s")
            response_options.append(formatted)
        for option in response_options:
            st.markdown(option)
        with st.form(key='selection_form'):
            selected_option = st.selectbox("✅ Choose Your Preferred Response:", options=response_options)
            comment = st.text_area("📝 Your Comment (optional):", value="", height=100)
            confirm_button = st.form_submit_button(label='✅ Confirm Selection')
        if confirm_button and selected_option:
            selected_idx = response_options.index(selected_option)
            chosen_response_info = st.session_state.generated_responses[selected_idx]
            chosen_mode = chosen_response_info.get("action_sequence", "Unknown")
            chosen_response_text = chosen_response_info.get("response", "")
            chosen_latency = chosen_response_info.get("Total Latency (s)", 0)
            update_conversation_history("🧑🏻 Todd", chosen_response_text)
            log_interaction(
                user_query=st.session_state.user_query,
                combined_system_prompt=st.session_state.combined_system_prompt,
                system_query=st.session_state.system_query,
                generated_responses=st.session_state.generated_responses,
                chosen_response={
                    "Mode": chosen_mode,
                    "Response": chosen_response_text,
                    "Total Latency (s)": chosen_latency,
                    "Comment": comment
                }
            )
            st.session_state.generated_responses = []
            st.success("✅ Response selected and added to conversation history.")
            if st.session_state.log_data:
                log_df = pd.DataFrame(st.session_state.log_data)
                save_logs_to_git(selected_user, log_df)
                st.info(f"Logs have been saved in the repository under 'KB_DIRECT/{st.session_state.username}/logs.csv'.")

# ------------------------ Sidebar for Downloading CSV ------------------------
st.sidebar.header("📥 Download Logs")
def download_logs():
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        df['Generated Responses'] = df['Generated Responses'].apply(lambda x: json.dumps(json.loads(x), ensure_ascii=False, indent=2))
        df['Chosen Response'] = df['Chosen Response'].apply(lambda x: json.dumps(x, ensure_ascii=False, indent=2))
        csv_data = df.to_csv(index=False)
        selected_user = st.session_state.get('username', 'user') or 'user'
        metric = st.session_state.get('metric', 'metric') or 'metric'
        est = pytz.timezone('US/Eastern')
        current_time = datetime.now(est)
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{selected_user}_{metric}_{time_str}.csv"
        st.sidebar.download_button(label="Download CSV Logs", data=csv_data, file_name=file_name, mime="text/csv")
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
            st.markdown(f"**{entry['speaker']}:** {entry['text']}")
        st.markdown('</div>', unsafe_allow_html=True)

