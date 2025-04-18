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
st.markdown("Interact with Todd and view your logs below.")

# Initialize session-specific log storage
if 'logs' not in st.session_state:
    st.session_state.logs = []

# CSV header for the session logs
csv_header = ["timestamp", "query", "response", "prepare_messages", "api_call", "process_response", "total"]

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# System prompt as provided
system_prompt = """
You are Todd, a natural, down-to-earth person. When you respond, use everyday language and avoid robotic or overly helpful phrases (like "I'm here to assist you" or "How can I help?"). Keep each answer under 15 words and sound genuine in your emotions. Do not mention that you are an AI assistant.

Assume you have a complete internal catalog of 30 conversation action sequences as follows:

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

When generating your responses, always append an annotation at the end of the "response" field in your JSON output that lists only the relevant conversation action sequence name (from the above catalog) that applies to the generated answer. The annotation should be enclosed in square brackets and list a single sequence name.
Return only the JSON output without any extra text.
"""

def clean_json_output(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def get_todd_response(user_query):
    # Create the user prompt based on the provided query
    user_prompt = f"""
For the query "{user_query}", perform the following steps:
1. Identify all the relevant conversation action sequences from the internal catalog above that apply to this query.
2. List these relevant action sequences.
3. For each identified action sequence, generate a diverse, conversational response as Todd, keeping each response under 15 words.
4. For each generated response, append an annotation at the end of the response text in square brackets that includes only the corresponding action sequence name.
Return the output in JSON format with the following structure:
{{
  "relevant_sequences": [ list of relevant action sequence names ],
  "responses": [
      {{
         "option_number": number,
         "response": "Generated answer text... [Action Sequence Name]"
      }},
      ...
  ]
}}
Return only the JSON output without any extra text.
"""
    total_start = time.time()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    time_messages = time.time() - total_start

    start_api = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Replace with your designated model name if needed
        messages=messages,
        temperature=0.7,
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

# Chat form: input query and display response
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your query:")
    submitted = st.form_submit_button("Submit")
    if submitted and user_query:
        with st.spinner("Processing your query..."):
            parsed_output, cleaned_output, timings = get_todd_response(user_query)
            
            st.subheader("Response from Todd")
            
            # Check if parsed output contains expected keys
            if "error" in parsed_output:
                st.error("Error processing response:")
                st.write(parsed_output["raw_output"])
            else:
                # Display relevant action sequences in colored boxes (deep blue with white text)
                if "relevant_sequences" in parsed_output:
                    st.markdown("### Relevant Action Sequences")
                    for seq in parsed_output["relevant_sequences"]:
                        st.markdown(
                            f"""<div style="background-color: #0056b3; padding: 8px; border-radius: 5px; margin-bottom: 5px;">
<strong style="color: white;">{seq}</strong>
</div>""", unsafe_allow_html=True)
                
                # Display each response with its corresponding action sequence
                if "responses" in parsed_output:
                    st.markdown("### Responses")
                    for resp in parsed_output["responses"]:
                        response_text = resp.get("response", "")
                        # Extract the annotation (action sequence) at the end enclosed in square brackets
                        match = re.search(r'\[(.*?)\]\s*$', response_text)
                        if match:
                            action_sequence = match.group(1)
                            # Remove the annotation from the response text
                            response_text_clean = re.sub(r'\s*\[.*?\]\s*$', '', response_text)
                        else:
                            action_sequence = "Unknown"
                            response_text_clean = response_text
                        
                        # Display action sequence header with a dark gray background
                        st.markdown(
                            f"""<div style="background-color: #343a40; padding: 8px; border-radius: 5px; margin-top: 10px;">
<strong style="color: white;">{action_sequence}</strong>
</div>""", unsafe_allow_html=True)
                        # Display response text in a light card with dark text
                        st.markdown(
                            f"""<div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin-bottom: 10px;">
<span style="color: #212529;">{response_text_clean}</span>
</div>""", unsafe_allow_html=True)
                
                # Display latency details in a contrasting box
                st.markdown("### Latency Details")
                latency_text = (
                    f"**Prepare Messages:** {timings['prepare_messages']:.3f} s  \n"
                    f"**API Call:** {timings['api_call']:.3f} s  \n"
                    f"**Process Response:** {timings['process_response']:.3f} s  \n"
                    f"**Total:** {timings['total']:.3f} s"
                )
                st.markdown(
                    f"""<div style="background-color: #004085; padding: 8px; border-radius: 5px; margin-top: 10px;">
<span style="color: #cce5ff;">{latency_text}</span>
</div>""", unsafe_allow_html=True)
            
            # Get current time in EST
            est = pytz.timezone('US/Eastern')
            timestamp = datetime.now(est).strftime("%Y-%m-%d %H:%M:%S")
            # Append the new log row to the session-specific logs
            new_row = [timestamp, user_query, cleaned_output, timings["prepare_messages"],
                       timings["api_call"], timings["process_response"], timings["total"]]
            st.session_state.logs.append(new_row)
            
            st.success("Query processed and logged!")

# Section to download the session-specific log as CSV
st.markdown("### Download Session Logs")
if st.session_state.logs:
    # Create an in-memory CSV file
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(csv_header)
    for row in st.session_state.logs:
        writer.writerow(row)
    csv_data = csv_buffer.getvalue()
    st.download_button(label="Download CSV Logs", data=csv_data, file_name="session_log.csv", mime="text/csv")
