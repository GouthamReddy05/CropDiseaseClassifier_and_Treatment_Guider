import os, requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

load_dotenv()
groq_api_key = os.getenv("groq_api_key")
app = Flask(__name__)

resp = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api_key}"
    },
    json={
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": "Hello from test script"}],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            if 'image' in request.files:
                ...
                return jsonify({
                    'prediction': prediction_text,
                    'knowledge': rag_output,
                    'llm_response': llm_result
                }), 200
            else:
                return jsonify({'error': 'No image uploaded.'}), 400
        else:
            data = request.get_json(force=True)
            user_message = data.get('message', '')
            if not user_message:
                return jsonify({'error': 'No message provided.'}), 400
            llm_result = groq_llm(user_message)
            return jsonify({'response': llm_result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


print(resp.status_code, resp.text)
