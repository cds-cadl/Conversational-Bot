import streamlit as st
import openai
import json
import re
import time
import csv
import io
from datetime import datetime
import pytz
import pandas as pd

# Set up the page
st.set_page_config(page_title="Action Sequences App", layout="wide")
st.title("Action Sequences App")
st.markdown("Interact with Todd and customize the prompt settings. View logs in History.")

# Define CSV header (used for logs)
csv_header = ["timestamp", "query", "response", "prepare_messages", "api_call", "process_response", "total"]

# Initialize session-specific log storage and prompt values
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'last_chat' not in st.session_state:
    st.session_state.last_chat = None

# Default prompt values
default_base_prompt = (
    "You are Todd, a natural, down-to-earth person. When you respond, use everyday language and avoid "
    "robotic or overly helpful phrases (like \"I'm here to assist you\" or \"How can I help?\"). Keep each answer "
    "under a fixed word limit and sound genuine in your emotions. Do not mention that you are an AI assistant."
)
default_additional_prompt = """Assume you have a complete internal catalog of 30 conversation action sequences as follows:

Greeting Sequence
Example: "Hey there!"

Introduction Sequence
Example: "Hi, I'm Todd."

Question and Answer Sequence
Example: "User: What time is it? Todd: It's 3 PM."

Request and Compliance/Refusal Sequence
Example: "User: Set an alarm for 7 AM. Todd: Can't do that, sorry."

Offer and Acceptance/Decline Sequence
Example: "Todd: Want to chat? User: Sure, why not."

Apology and Acceptance Sequence
Example: "Todd: My bad. User: No worries."

Complaint and Response Sequence
Example: "User: This is frustrating. Todd: I get it."

Compliment and Acknowledgment Sequence
Example: "User: You're cool! Todd: Thanks, appreciate it."

Invitation and Acceptance/Decline Sequence
Example: "Todd: Join me for a coffee? User: Sounds good."

Farewell Sequence
Example: "User: Bye! Todd: Catch you later!"

Confirmation and Acknowledgment Sequence
Example: "User: Is that right? Todd: Yup, that's it."

Statement and Agreement/Disagreement Sequence
Example: "User: I think it's true. Todd: I agree."

Suggestion and Acceptance/Refusal Sequence
Example: "Todd: How about a walk? User: Nice idea."

Clarification Request and Explanation Sequence
Example: "Todd: What do you mean? User: Like, explain please."

Announcement and Reaction Sequence
Example: "User: I got a job! Todd: That's awesome!"

Problem Statement and Solution Offer Sequence
Example: "User: I'm stuck. Todd: Maybe try a break."

Expression of Emotion and Support Sequence
Example: "User: I'm upset. Todd: That sucks, hang in there."

Interruption and Permission Sequence
Example: "Todd: Mind if I jump in? User: Go ahead."

Correction and Acknowledgment Sequence
Example: "User: It's 5, not 4. Todd: Oh, thanks for the heads up."

Topic Shift and Acceptance Sequence
Example: "User: Let's change the subject. Todd: Sure, what's next?"

Summons and Response Sequence
Example: "User: Hey, Todd! Todd: Yes?"

Expressing Uncertainty and Reassurance Sequence
Example: "User: I'm not sure. Todd: It's alright, you'll figure it out."

Giving Directions and Acknowledgment Sequence
Example: "Todd: Turn left at the street. User: Got it."

Offering Assistance and Acceptance/Decline Sequence
Example: "Todd: Need a hand? User: No, I'm fine."

Expressing Doubt and Clarification Sequence
Example: "User: That doesn't sound right. Todd: What do you mean?"

Agreement and Extension Sequence
Example: "User: That makes sense. Todd: Plus, there's more to it."

Refusal and Justification Sequence
Example: "User: Do this for me. Todd: Sorry, can't do that."

Reminder and Acknowledgment Sequence
Example: "User: Remind me later. Todd: Sure, will do."

Expression of Surprise and Explanation Sequence
Example: "User: Wow, really? Todd: Yeah, it's surprising!"

Permission Request and Grant/Denial Sequence
Example: "Todd: Mind if I share? User: Go ahead."
"""
default_query_prompt_template = """For the query "{user_query}", perform the following steps:
1. Identify all the relevant conversation action sequences from the internal catalog above that apply to this query.
2. List these relevant action sequences.
3. For each identified action sequence, generate 2 diverse, conversational responses as Todd, keeping each response under 15 words.
4. For each generated response, append an annotation at the end of the response text in square brackets that includes only the corresponding action sequence name.
Return the output in JSON format with the following structure:
{{
  "relevant_sequences": [ list of relevant action sequence names ],
  "responses": [
      {{
         "action_sequence": "Name of action sequence",
         "option_number": number,
         "response": "Generated answer text... [Action Sequence Name]"
      }},
      ...
  ]
}}
Return only the JSON output without any extra text.
"""

# Store editable prompt values in session state if not already there
if 'base_prompt' not in st.session_state:
    st.session_state.base_prompt = default_base_prompt
if 'additional_prompt' not in st.session_state:
    st.session_state.additional_prompt = default_additional_prompt
if 'query_prompt_template' not in st.session_state:
    st.session_state.query_prompt_template = default_query_prompt_template

# ---------------------
# Editable Prompt Update Forms
# ---------------------
st.markdown("## Update Prompt Settings")

with st.expander("Edit Base Prompt (Hidden by default)"):
    with st.form("update_base_prompt"):
        updated_base = st.text_area("Base Prompt", value=st.session_state.base_prompt, height=150)
        if st.form_submit_button("Update Base Prompt"):
            st.session_state.base_prompt = updated_base
            st.success("Base Prompt updated.")

with st.form("update_additional_prompt"):
    updated_additional = st.text_area("Additional Prompt (Action Sequences)", 
                                      value=st.session_state.additional_prompt, height=300)
    if st.form_submit_button("Update Additional Prompt"):
        st.session_state.additional_prompt = updated_additional
        st.success("Additional Prompt updated.")

with st.form("update_query_template"):
    updated_query_template = st.text_area("Query Prompt Template", 
                                          value=st.session_state.query_prompt_template, height=300)
    if st.form_submit_button("Update Query Prompt Template"):
        st.session_state.query_prompt_template = updated_query_template
        st.success("Query Prompt Template updated.")

# ---------------------
# Display the current Base Prompt (read-only display)
st.markdown("### Current Base Prompt")
st.text_area("Base Prompt (Current)", value=st.session_state.base_prompt, height=150, disabled=True)

# ---------------------
# Temperature slider
st.markdown("### Response Temperature")
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# ---------------------
# Helper Functions
def clean_json_output(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def get_todd_response(user_query, temperature):
    # Combine system prompt: current base prompt + additional prompt from session state
    combined_system_prompt = st.session_state.base_prompt + "\n\n" + st.session_state.additional_prompt

    # Build the user prompt using the template from session state
    user_prompt = st.session_state.query_prompt_template.format(user_query=user_query)
    
    total_start = time.time()
    messages = [
        {"role": "system", "content": combined_system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    time_messages = time.time() - total_start

    start_api = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Replace with your designated model if needed
        messages=messages,
        temperature=temperature,
        max_tokens=700,
    )
    time_api = time.time() - start_api

    start_process = time.time()
    output = response.choices[0].message['content']
    time_process = time.time() - start_process
    total_time = time.time() - total_start

    cleaned_output = clean_json_output(output)
    try:
        parsed_output = json.loads(cleaned_output)
    except json.JSONDecodeError:
        parsed_output = {"error": "JSON decode error", "raw_output": cleaned_output}
    
    timings = {
        "prepare_messages": time_messages,
        "api_call": time_api,
        "process_response": time_process,
        "total": total_time
    }
    return parsed_output, cleaned_output, timings

# ---------------------
# Main Chat Interface
tabs = st.tabs(["Chat", "History"])

with tabs[0]:
    st.subheader("Chat here")
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Enter your query:")
        submitted = st.form_submit_button("Submit")
        if submitted and user_query:
            with st.spinner("Processing your query..."):
                parsed_output, cleaned_output, timings = get_todd_response(user_query, temperature)
                
                chat_content = f"**Your Query:** {user_query}\n\n"
                
                if "error" in parsed_output:
                    chat_content += f"**Error processing response:** {parsed_output['raw_output']}\n\n"
                else:
                    if "relevant_sequences" in parsed_output:
                        chat_content += "**Relevant Action Sequences:**\n"
                        for seq in parsed_output["relevant_sequences"]:
                            chat_content += f"- {seq}\n"
                        chat_content += "\n"
                    
                    if "responses" in parsed_output:
                        chat_content += "**Responses:**\n"
                        for resp in parsed_output["responses"]:
                            action_sequence = resp.get("action_sequence", "Unknown")
                            response_text = resp.get("response", "")
                            response_text_clean = re.sub(r'\s*\[.*?\]\s*$', '', response_text)
                            chat_content += f"- **{action_sequence}:** {response_text_clean}\n"
                    
                    latency_text = (
                        f"Prepare Messages: {timings['prepare_messages']:.3f} s  |  "
                        f"API Call: {timings['api_call']:.3f} s  |  "
                        f"Process Response: {timings['process_response']:.3f} s  |  "
                        f"Total: {timings['total']:.3f} s"
                    )
                    chat_content += f"\n**Latency Details:** {latency_text}\n"
                
                est = pytz.timezone('US/Eastern')
                timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S")
                new_row = [timestamp, user_query, cleaned_output, timings["prepare_messages"],
                           timings["api_call"], timings["process_response"], timings["total"]]
                st.session_state.logs.append(new_row)
                st.session_state.last_chat = chat_content
                
                st.success("Query processed and logged!")
    
    if st.session_state.last_chat:
        st.markdown("### Last Chat")
        st.markdown(st.session_state.last_chat)

with tabs[1]:
    st.subheader("Chat History")
    if st.session_state.logs:
        df_logs = pd.DataFrame(st.session_state.logs, columns=csv_header)
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.info("No chat history yet.")

st.markdown("### Download Session Logs")
if st.session_state.logs:
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(csv_header)
    for row in st.session_state.logs:
        writer.writerow(row)
    csv_data = csv_buffer.getvalue()
    st.download_button(label="Download CSV Logs", data=csv_data, file_name="session_log.csv", mime="text/csv")

