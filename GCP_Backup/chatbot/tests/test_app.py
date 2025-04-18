# tests/test_app.py

import pytest
import sys
import os
import json
from unittest.mock import MagicMock
from datetime import datetime

# Adjust the path to import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app

def test_parse_responses_valid_json():
    raw_response = '''
    [
        {"option_number": 1, "response": "Response one."},
        {"option_number": 2, "response": "Response two."}
    ]
    '''
    expected_output = [
        {"Option Number": 1, "Response Text": "Response one."},
        {"Option Number": 2, "Response Text": "Response two."}
    ]
    assert app.parse_responses(raw_response) == expected_output

def test_parse_responses_invalid_json():
    raw_response = "This is not a JSON array."
    expected_output = [{"Option Number": "N/A", "Response Text": "This is not a JSON array."}]
    assert app.parse_responses(raw_response) == expected_output

def test_parse_responses_partial_json():
    raw_response = '''
    Some introductory text.
    [
        {"option_number": 1, "response": "Response one."},
        {"option_number": 2, "response": "Response two."}
    ]
    Some concluding text.
    '''
    expected_output = [
        {"Option Number": 1, "Response Text": "Response one."},
        {"Option Number": 2, "Response Text": "Response two."}
    ]
    assert app.parse_responses(raw_response) == expected_output

def test_parse_responses_empty_response():
    raw_response = ''
    expected_output = [{"Option Number": "N/A", "Response Text": ""}]
    assert app.parse_responses(raw_response) == expected_output

def test_parse_responses_non_list_json():
    raw_response = '{"option_number": 1, "response": "Response one."}'
    expected_output = [{"Option Number": "N/A", "Response Text": '{"option_number": 1, "response": "Response one."}'}]
    assert app.parse_responses(raw_response) == expected_output

def test_log_interaction(monkeypatch):
    # Mock Streamlit's session_state
    mock_session_state = {
        'username': 'Jeff',
        'metric': 'Sales',
        'log_data': []
    }

    def mock_get(key, default):
        return mock_session_state.get(key, default)

    monkeypatch.setattr('app.st.session_state.get', mock_get)
    monkeypatch.setattr('app.st.session_state', mock_session_state)

    user_query = "Tell me what caused this problem."
    combined_system_prompt = "Combined system prompt."
    system_query = "System query."
    generated_responses = [
        {"Mode": "naive", "Response": "Response one.", "Latency (s)": 0.5},
        {"Mode": "hybrid", "Response": "Response two.", "Latency (s)": 0.7}
    ]
    chosen_response = {
        "Mode": "naive",
        "Response": "Response one.",
        "Latency (s)": 0.5,
        "Comment": "This is helpful."
    }

    app.log_interaction(user_query, combined_system_prompt, system_query, generated_responses, chosen_response)

    assert len(mock_session_state['log_data']) == 1
    log_entry = mock_session_state['log_data'][0]
    assert log_entry['Username'] == 'Jeff'
    assert log_entry['Metric'] == 'Sales'
    assert log_entry['User Query'] == user_query
    assert log_entry['Combined System Prompt'] == combined_system_prompt
    assert log_entry['System Query'] == system_query
    assert log_entry['Generated Responses'] == json.dumps(generated_responses, ensure_ascii=False)
    assert log_entry['Chosen Response'] == json.dumps(chosen_response, ensure_ascii=False)
    assert 'Timestamp' in log_entry
