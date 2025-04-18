from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
import logging
import uvicorn
import csv
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime
from dateutil.parser import parse
import os
import pytz
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

executor = ThreadPoolExecutor(max_workers=10)
headers = {"Content-Type": "application/json"}

# Directory to store session CSV files
session_csv_dir = "session_csv_files"
os.makedirs(session_csv_dir, exist_ok=True)

# In-memory history of the conversation (last 3 prompts and responses)
conversation_history = []
full_conversation_history = []
csv_file_path = None
time_responses_sent = None
time_chosen_response_received = None

# Eastern Time zone with DST handling (Eastern Time with DST awareness)
ET = pytz.timezone('US/Eastern')

# Set up the language model
openai_api_key = "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"  # Replace with your actual OpenAI API key
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

# Generate a unique filename for each session
def generate_csv_filename():
    timestamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    return os.path.join(session_csv_dir, f"conversation_history_{timestamp}.csv")

def check_last_entry(history):
    if history and history[-1][1] is None:
        logging.warning("Incomplete entry found in conversation history.")
        return handle_incomplete_entry(history)
    return None

def handle_incomplete_entry(history):
    incomplete_entry = history.pop()
    return f"Didn't choose a response; removed: {incomplete_entry[0]}"

def initialize_csv_file(path):
    try:
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'date_time', 'prompt', 'history', 'responses', 'chosen_response', 'emotion', 'personalize', 'server_to_pi_latency', 'pi_processing_latency', 'pi_to_server_latency', 'api_latency', 'chosen_response_latency'])
    except Exception as e:
        logging.error(f"Failed to create CSV: {e}")

def append_to_csv_file(path, entry):
    try:
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            (index, date_time, partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize, server_to_pi_latency, pi_processing_latency, pi_to_server_latency, api_latency, chosen_response_latency) = entry
            responses_list = model_responses.get('responses', [])
            history_str = ";\n".join(["\n".join(map(str, pair)) for pair in (history_snapshot or [])])
            responses_str = ';\n'.join(map(str, responses_list))
            writer.writerow([str(index), date_time, str(partner_prompt), history_str, responses_str, str(user_response), str(emotion), str(personalize), str(server_to_pi_latency), str(pi_processing_latency), str(pi_to_server_latency), str(api_latency), str(chosen_response_latency)])
    except Exception as e:
        logging.error(f"Failed to append to CSV: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global csv_file_path, conversation_history, full_conversation_history
    csv_file_path = generate_csv_filename()
    conversation_history = []
    full_conversation_history = []
    initialize_csv_file(csv_file_path)

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                time_received_osdpi = datetime.now(ET)
                print(f'Data received from OS-DPI at {time_received_osdpi}')
            try:
                data_json = json.loads(data)
                state = data_json.get("state", {})
                prefix = state.get("$prefix", "")
                emotion = state.get("$Style", "")
                personalize = state.get("$personalize", None)

                if prefix == 'prompt':
                    incomplete_message = check_last_entry(conversation_history)
                    
                    # Extract prompt from previous context
                    prompt = state.get("$prompt", "")

                    # Prepare and send the prompt for response generation
                    if prompt:
                        print('inside prompt if')
                        logging.info(f"Received prompt: {prompt} | Emotion: {emotion} | Personalize: {personalize}")
                        loop = asyncio.get_running_loop()

                        # Generate the context from the last three conversation turns
                        context = "\n".join(["\n".join(map(str, pair)) for pair in conversation_history[-3:] if pair[1] is not None])

                        print(f"Context: {context}")
                        logging.info(f"Sending full prompt to model: {prompt}")
                        
                        api_request_start_time = datetime.now(ET)
                        response = await loop.run_in_executor(executor, regenerate_responses, prompt, emotion, context)
                        api_request_end_time = datetime.now(ET)
                        
                        api_latency = (api_request_end_time - api_request_start_time).total_seconds()

                        responses_dict = {f"response{i+1}": resp for i, resp in enumerate(response)}
                        responses_dict['Display'] = prompt
                        if incomplete_message:
                            responses_dict['warning'] = incomplete_message
                        print(responses_dict)
                        time_responses_sent = datetime.now(ET)
                        await websocket.send_text(json.dumps(responses_dict))

                        update_history(conversation_history, prompt, None, response, full_conversation_history, emotion, personalize)
                        
                    else:
                        logging.error("No prompt found in the received data.")
                elif prefix == 'Chosen':
                    chosen_response = state.get("$socket", "")
                    time_chosen_response_received = datetime.now(ET)
                    chosen_response_latency = (time_chosen_response_received - time_responses_sent).total_seconds()

                    if chosen_response:
                        logging.info(f"Received chosen response: {chosen_response}")
                        if conversation_history and conversation_history[-1][1] is None:
                            conversation_history[-1] = (conversation_history[-1][0], chosen_response)
                            update_full_history(full_conversation_history, conversation_history[-1], chosen_response)
                            timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
                            append_to_csv_file(csv_file_path, (len(conversation_history), timestamp, conversation_history[-1][0], full_conversation_history[-1][1], chosen_response, conversation_history[-4:-1], full_conversation_history[-1][4], full_conversation_history[-1][5], None, None, None, api_latency, chosen_response_latency))
                        else:
                            logging.error("Chosen response received without a corresponding prompt.")
                    else:
                        logging.error("No chosen response found in the received data.")
                
                elif prefix == 'regenerate':
                    prompt = state.get("$prompt", "")
                    if not prompt:
                        logging.error("No prompt provided for regeneration.")
                        continue

                    logging.info(f"Regenerating responses for prompt: {prompt} | Emotion: {emotion}")

                    # Generate the context from the last three conversation turns
                    context = "\n".join(["\n".join(map(str, pair)) for pair in conversation_history[-3:] if pair[1] is not None])

                    responses_list = await regenerate_responses(prompt, emotion, context)
                    responses_dict = {f"response{i+1}": resp for i, resp in enumerate(responses_list)}
                    responses_dict['Display'] = prompt

                    # Update the conversation history
                    if conversation_history and conversation_history[-1][0] == prompt:
                        conversation_history[-1] = (prompt, None)
                    else:
                        conversation_history.append((prompt, None))

                    time_responses_sent = datetime.now(ET)
                    await websocket.send_text(json.dumps(responses_dict))

                elif prefix == 'new_conv':
                    logging.info("Received new_conv prefix, clearing conversation history and starting new conversation.")
                    conversation_history.clear()

                else:
                    logging.error(f"Unexpected prefix value: {prefix}")

            except json.JSONDecodeError:
                logging.error("Invalid JSON received.")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")

def regenerate_responses(prompt, emotion, context):
    # Define the set of allowed emotions
    allowed_emotions = {"cheerful", "sad", "terrified"}

    # Check if the provided emotion is in the allowed set; if not, default to Positive/Negative
    if emotion not in allowed_emotions:
        emotion_1 = "Positive"
        emotion_2 = "Negative"
    else:
        emotion_1 = emotion
        emotion_2 = emotion

    responses = []

    # Define base template
    base_template = f"\
    Use the following pieces of context to override the conversation reply truthfully and use the emotion tone.\
    If the context does not provide the truthful answer, make the answer as truthful as possible. You are answering as the User.\
    Keep the response as concise as possible.\
    Context: {{context}}\
    Question: {{question}}\
    Response: {{response}}.\
    Emotion Tone: {{emotion_tone}}\
    Keep it strictly within this number of words: {{length_words}}\
    Truthful Response:"

    for i in range(2):
        # First two responses with standard length (e.g., 15 words)
        emotion_tone = emotion_1 if i % 2 == 0 else emotion_2
        template = PromptTemplate(
            input_variables=["context", "question", "response", "emotion_tone", "length_words"],
            template=base_template
        )
        prompt_with_template = template.format(
            context=context,
            question=prompt,
            response="Generated response",
            emotion_tone=emotion_tone,
            length_words=15,
        )
        response = llm(prompt_with_template)
        responses.append(response.content.strip())

    for i in range(2):
        # Next two responses with shorter length (e.g., 7 words)
        emotion_tone = emotion_1 if i % 2 == 0 else emotion_2
        template = PromptTemplate(
            input_variables=["context", "question", "response", "emotion_tone", "length_words"],
            template=base_template
        )
        prompt_with_template = template.format(
            context=context,
            question=prompt,
            response="Generated response",
            emotion_tone=emotion_tone,
            length_words=7,
        )
        response = llm(prompt_with_template)
        responses.append(response.content.strip())

    return responses

def update_history(history, partner_prompt, user_response, model_responses, full_history, emotion, personalize):
    history_snapshot = history[-3:]
    while len(history) > 3:
        history.pop(0)
    history.append((partner_prompt, user_response))
    if model_responses is not None:
        full_history.append((partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize))

def update_full_history(full_history, last_convo_pair, chosen_response):
    for index, (partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize) in enumerate(full_history):
        if partner_prompt == last_convo_pair[0] and user_response is None:
            full_history[index] = (partner_prompt, model_responses, chosen_response, history_snapshot, emotion, personalize)
            break

@app.get("/download_csv")
async def download_csv():
    global csv_file_path
    try:
        return FileResponse(csv_file_path, media_type='text/csv', filename=os.path.basename(csv_file_path))
    except Exception as e:
        logging.error(f"Failed to generate CSV: {e}")
        return {"error": f"Failed to generate CSV: {e}"}

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5678, log_level="info")
