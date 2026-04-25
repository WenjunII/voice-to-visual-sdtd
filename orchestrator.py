import os
import time
import requests
import json
from python_osc import udp_client

# --- Configuration ---
OSC_IP = "127.0.0.1"
OSC_PORT = 7000
CAPRIOLE_API_KEY = "sk-ul90SZYJfLC3V8C-PXMaeCwT0X-bCkrDDebiHFh0V4c"
CAPRIOLE_ENDPOINT = "https://api.caprioletech.com/v1/chat"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# The sequence of models to attempt (as a fallback chain)
MODEL_SEQUENCE = [
    {"type": "capriole", "model": "anthropic/claude-opus-4-7-thinking"},
    {"type": "capriole", "model": "google/gemini-3.1-pro-preview"},
    {"type": "capriole", "model": "openai/gpt-5.5"},
    {"type": "ollama", "model": "gemini-3-flash-preview"},
    {"type": "ollama", "model": "kimi-k2.6:cloud"}
]

class MultiLLMClient:
    def __init__(self):
        self.api_key = CAPRIOLE_API_KEY

    def generate_prompt(self, raw_speech, scene_context):
        system_instruction = (
            "You are a Visual Prompt Engineer for SDXL. "
            f"Current Scene Context: {scene_context}. "
            "Task: Convert the new speech input into a high-quality, descriptive SDXL prompt under 70 tokens. "
            "Output ONLY the prompt string."
        )
        
        full_input = f"{system_instruction}\n\nNew Speech Input: {raw_speech}"

        for entry in MODEL_SEQUENCE:
            try:
                print(f"[TRYING MODEL]: {entry['model']}...")
                if entry["type"] == "capriole":
                    result = self._call_capriole(entry["model"], full_input)
                else:
                    result = self._call_ollama(entry["model"], full_input)
                
                if result:
                    return result.strip()
            except Exception as e:
                print(f"Error with {entry['model']}: {e}")
                continue
        
        return None

    def _call_capriole(self, model_name, input_text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "input": input_text
        }
        response = requests.post(CAPRIOLE_ENDPOINT, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            # Assuming the response format follows the input structure or similar
            # Based on common patterns, let's try to extract text
            try:
                # Based on the user's curl, it's a bit ambiguous what the response body is.
                # Usually these proxies return a simple JSON with a 'response' or 'output' field.
                res_json = response.json()
                return res_json.get("output") or res_json.get("response") or res_json.get("choices", [{}])[0].get("message", {}).get("content")
            except:
                return response.text
        return None

    def _call_ollama(self, model_name, input_text):
        data = {
            "model": model_name,
            "prompt": input_text,
            "stream": False
        }
        response = requests.post(OLLAMA_ENDPOINT, json=data, timeout=15)
        if response.status_code == 200:
            return response.json().get("response")
        return None

class PromptOrchestrator:
    def __init__(self):
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        self.llm_client = MultiLLMClient()
        self.last_prompt = ""
        
    def refine_and_send(self, raw_text):
        if not raw_text.strip():
            return

        print(f"\n[WHISPER]: {raw_text}")
        
        refined_prompt = self.llm_client.generate_prompt(raw_text, self.last_prompt)
        
        if not refined_prompt:
            print("Fallback: Using raw text (All models failed)")
            refined_prompt = f"{raw_text}, cinematic, 8k"

        if refined_prompt != self.last_prompt:
            print(f"[REFINED PROMPT]: {refined_prompt}")
            print(f"[OSC SEND]: /prompt")
            self.osc_client.send_message("/prompt", refined_prompt)
            self.last_prompt = refined_prompt

if __name__ == "__main__":
    orch = PromptOrchestrator()
    orch.refine_and_send("a futuristic space station orbiting a dying star")
