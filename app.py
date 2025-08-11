from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from python_files.main import predict_disease, process_image, encode_output
from python_files.rag import run_rag_pipeline

load_dotenv()

app = Flask(__name__)

# Load Groq API key from .env
groq_api_key = os.getenv('GROQ_API_KEY')  # Make sure your .env has GROQ_API_KEY=your_key

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def groq_llm(prompt):
    """Send prompt to Groq API and return the generated response."""
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )

        if not response.ok:
            return f"Error: {response.status_code} - {response.text}"

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Error communicating with Groq API: {e}")
        return "Error communicating with Groq API."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handles text chat requests."""
    if not request.is_json:
        return jsonify({'error': 'Expected JSON format'}), 400

    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    llm_result = groq_llm(user_message)
    return jsonify({'reply': llm_result})


@app.route('/process_image', methods=['POST'])
def process_image_api():
    """Handles image upload and disease prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    os.makedirs('uploads', exist_ok=True)
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    processed_image_path = process_image(image_path)
    prediction = predict_disease(processed_image_path)
    prediction_text = f"Predicted Disease: {prediction}"

    rag_output = run_rag_pipeline(prediction_text)

    combined_prompt = (
        f"Here is the disease prediction: {prediction_text}\n\n"
        f"Additional information from knowledge base: {rag_output}\n\n"
        f"Based on the above, give a detailed explanation and treatment plan."
    )

    llm_result = groq_llm(combined_prompt)

    return jsonify({
        'prediction': prediction_text,
        'knowledge': rag_output,
        'reply': llm_result
    })


if __name__ == '__main__':
    app.run(debug=True)
