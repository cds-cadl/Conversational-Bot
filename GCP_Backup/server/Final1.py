from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
import logging
import uvicorn
import csv
from concurrent.futures import ThreadPoolExecutor
import requests
import asyncio
from datetime import datetime
from dateutil.parser import parse
import os

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

executor = ThreadPoolExecutor(max_workers=10)
api_url = "http://localhost:7000/generate_response_informed"  # MAKE SURE THIS IS LOCALHOST AND NOT THE EXTERNAL IP
rasp_pi_api_url = "https://humane-marmot-entirely.ngrok-free.app/"  # UPDATE WHEN GETTING A NEW SOCKETXP URL
headers = {"Content-Type": "application/json"}

# Directory to store session CSV files
session_csv_dir = "session_csv_files"
os.makedirs(session_csv_dir, exist_ok=True)

# In-memory history of the conversation (last 3 prompts and responses)
conversation_history = []
full_conversation_history = []
csv_file_path = None

# Generate a unique filename for each session
def generate_csv_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(session_csv_dir, f"conversation_history_{timestamp}.csv")

def get_speech_to_text():
    data = requests.get(f'{rasp_pi_api_url}/get_audio_transcription')
    data_json = data.json()
    return data_json

def check_last_entry(history):
    # Check if the last entry in the history has a chosen response
    if history and history[-1][1] is None:
        # Handle the incomplete entry
        logging.warning("Incomplete entry found in conversation history.")
        return handle_incomplete_entry(history)
    return None

def handle_incomplete_entry(history):
    # Remove the incomplete entry and return a message
    incomplete_entry = history.pop()
    return f"Didn't choose a response; removed: {incomplete_entry[0]}"

def initialize_csv_file(path):
    try:
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'date_time', 'prompt', 'history', 'responses', 'chosen_response', 'emotion', 'personalize'])
    except Exception as e:
        logging.error(f"Failed to create CSV: {e}")

def append_to_csv_file(path, entry):
    try:
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            (index, date_time, partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize) = entry
            responses_list = model_responses.get('responses', [])
            history_str = ";\n".join(["\n".join(map(str, pair)) for pair in (history_snapshot or [])])
            responses_str = ';\n'.join(map(str, responses_list))
            writer.writerow([str(index), date_time, str(partner_prompt), history_str, responses_str, str(user_response), str(emotion), str(personalize)])
    except Exception as e:
        logging.error(f"Failed to append to CSV: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global csv_file_path, conversation_history, full_conversation_history
    csv_file_path = generate_csv_filename()  # Generate new filename for this session
    conversation_history = []
    full_conversation_history = []
    initialize_csv_file(csv_file_path)  # Initialize the CSV file

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                time_received_osdpi = datetime.now()
                print(f'Data received from OS-DPI at {time_received_osdpi}')
            try:
                data_json = json.loads(data)
                state = data_json.get("state", {})
                prefix = state.get("$prefix", "")
                emotion = state.get("$Style", "")
                personalize = state.get("$personalize", None)  # Extracting $personalize value

                if prefix == 'prompt':
                    incomplete_message = check_last_entry(conversation_history)
                    
                    time_server_sent_to_rasp_pi = datetime.now()
                    rasp_pi_data = get_speech_to_text()
                    time_server_received_from_rasp_pi = datetime.now()
                    prompt = rasp_pi_data['text']
                    time_rasp_pi_received_from_server = parse(rasp_pi_data['time_received'])
                    time_rasp_pi_sent_to_server = parse(rasp_pi_data['time_processed'])
                    time_rasp_pi_processing = rasp_pi_data['total_time']
                    print('prompt is ', prompt)
                    message = json.dumps({'state': {"$Display": prompt}})
                    print(f'time_server_sent_to_rasp_pi - {time_server_sent_to_rasp_pi}')
                    print(f'time_server_received_from_rasp_pi - {time_server_received_from_rasp_pi}')
                    print(f'time_rasp_pi_received_from_server - {time_rasp_pi_received_from_server}')
                    print(f'time_rasp_pi_sent_to_server - {time_rasp_pi_sent_to_server}')

                    print(f'time taken for request to reach rasp pi from server - {str(time_rasp_pi_received_from_server - time_server_sent_to_rasp_pi)}')
                    print(f'time taken rasp pi and nova to process request - {time_rasp_pi_processing}')
                    print(f'time taken for request to reach server from rasp pi - {str(time_server_received_from_rasp_pi - time_rasp_pi_sent_to_server)}')

                    print(f'total time taken for request to leave and reach server - {str(time_server_received_from_rasp_pi - time_server_sent_to_rasp_pi)}')

                    await websocket.send_text(message)

                    if prompt:
                        print('inside prompt if')
                        logging.info(f"Received prompt: {prompt} | Emotion: {emotion} | Personalize: {personalize}")
                        loop = asyncio.get_running_loop()

                        # Prepare the full prompt by concatenating the last 3 history pairs with the new prompt
                        full_prompt = "\n".join(["\n".join(map(str, pair)) for pair in conversation_history[-3:] if pair[1] is not None]) + "\n" + prompt
                        print(full_prompt)
                        logging.info(f"Sending full prompt to model: {full_prompt}")
                        response = await loop.run_in_executor(executor, send_to_api_sync, full_prompt, emotion, personalize)

                        # Assuming 'responses' is a key in the JSON data that contains the list you're interested in
                        responses_list = response.get('responses', [])

                        # Now create your dictionary
                        responses_dict = {f"response{i+1}": resp for i, resp in enumerate(responses_list)}
                        responses_dict['Display'] = prompt
                        if incomplete_message:
                            responses_dict['warning'] = incomplete_message  # Add the log message about incomplete entry
                        print(responses_dict)
                        await websocket.send_text(json.dumps(responses_dict))

                        # Update the conversation history with the prompt and placeholder for the user's chosen response
                        update_history(conversation_history, prompt, None, response, full_conversation_history, emotion, personalize)
                    else:
                        logging.error("No prompt found in the received data.")
                elif prefix == 'Chosen':
                    chosen_response = state.get("$socket", "")
                    if chosen_response:
                        logging.info(f"Received chosen response: {chosen_response}")
                        # Update the last entry in the conversation history with the chosen response
                        if conversation_history and conversation_history[-1][1] is None:
                            conversation_history[-1] = (conversation_history[-1][0], chosen_response)
                            update_full_history(full_conversation_history, conversation_history[-1], chosen_response)
                            # Append to CSV file with the chosen response
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            append_to_csv_file(csv_file_path, (len(conversation_history), timestamp, conversation_history[-1][0], full_conversation_history[-1][1], chosen_response, conversation_history[-4:-1], full_conversation_history[-1][4], full_conversation_history[-1][5]))
                        else:
                            logging.error("Chosen response received without a corresponding prompt.")
                    else:
                        logging.error("No chosen response found in the received data.")

                elif prefix == 'new_conv':
                    logging.info("Received new_conv prefix, clearing conversation history and starting new conversation.")
                    # Clear the conversation history
                    conversation_history.clear()

                else:
                    logging.error(f"Unexpected prefix value: {prefix}")

            except json.JSONDecodeError:
                logging.error("Invalid JSON received.")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")

def update_history(history, partner_prompt, user_response, model_responses, full_history, emotion, personalize):
    history_snapshot = history[-3:]
    # Trim the history to only keep the last 3 conversation pairs
    while len(history) > 3:
        history.pop(0)

    # Update the in-memory history with the partner's prompt and a placeholder for the user's response
    history.append((partner_prompt, user_response))

    # Update the full conversation history with the new conversation pair, model responses, emotion, and personalize
    if model_responses is not None:
        full_history.append((partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize))

def update_full_history(full_history, last_convo_pair, chosen_response):
    # Find the last conversation pair in the full history and update it with the chosen response
    for index, (partner_prompt, model_responses, user_response, history_snapshot, emotion, personalize) in enumerate(full_history):
        if partner_prompt == last_convo_pair[0] and user_response is None:
            full_history[index] = (partner_prompt, model_responses, chosen_response, history_snapshot, emotion, personalize)
            break

@app.get("/download_csv")
async def download_csv():
    global csv_file_path
    try:
        # After generating, return the file
        return FileResponse(csv_file_path, media_type='text/csv', filename=os.path.basename(csv_file_path))
    except Exception as e:
        logging.error(f"Failed to generate CSV: {e}")
        return {"error": f"Failed to generate CSV: {e}"}

def send_to_api_sync(prompt, emotion, personalize):
    try:
        payload = {'prompt': prompt, 'emotion': emotion, 'personalize': personalize}
        response = requests.post(api_url, headers=headers, json=payload)
        logging.info(f"Sending payload to API: {payload}")
        response.raise_for_status()
        print(response)
        return response.json()
    except requests.HTTPError as e:
        logging.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return []
    except requests.RequestException as e:
        logging.error(f"Error sending request to API: {e}")
        return []

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5678, log_level="info")

