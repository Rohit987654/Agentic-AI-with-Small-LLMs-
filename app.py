from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS  # allows frontend and backend to communicate

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can connect

# Load TinyLlama model safely for CPU or GPU
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

# Define chatbot function
def resolve_ticket(message):
    prompt = f"""
You are a helpful support agent resolving customer tickets.

Ticket:
{message}

Resolution:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Resolution:")[-1].strip()


@app.route("/", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("data", [""])[0]
    reply = resolve_ticket(message)
    return jsonify({"data": [reply]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860)
