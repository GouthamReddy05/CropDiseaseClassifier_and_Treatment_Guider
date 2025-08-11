from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from python_files.main import predict_disease, process_image, encode_output
from python_files.rag import run_rag_pipeline

load_dotenv()

app = Flask(__name__)

# Load Groq API key (ensure your .env file has GROQ_API_KEY=...)
groq_api_key = os.getenv('GROQ_API_KEY')

# Groq API endpoint for chat completions
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_llm(prompt):
    """Send prompt to Groq API and get the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api_key}"
    }
    data = {
        "model": "llama-3.1-70b-versatile",  # You can change this model if needed
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        if response.ok:
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return f"Unexpected API response: {result}"
        else:
            return f"Groq API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error communicating with Groq API: {e}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Handle file upload (image)
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join('uploads', image_file.filename)
            os.makedirs('uploads', exist_ok=True)
            image_file.save(image_path)

            # Process and predict
            processed_image_path = process_image(image_path)
            prediction = predict_disease(processed_image_path)
            prediction_text = f"Predicted Disease: {prediction}"

            # Encode + retrieve knowledge
            encoded_response = encode_output(prediction_text)
            rag_output = run_rag_pipeline(prediction_text)

            # Build prompt for Groq LLM
            combined_prompt = (
                f"Here is the disease prediction: {prediction_text}\n\n"
                f"Additional information from knowledge base: {rag_output}\n\n"
                f"Based on the above, give a detailed explanation and treatment plan."
            )

            llm_result = groq_llm(combined_prompt)

            # Cleanup
            os.remove(processed_image_path)

            return jsonify({
                'prediction': prediction_text,
                'knowledge': rag_output,
                'llm_response': llm_result
            })
        else:
            return jsonify({'response': 'No image uploaded.'})

    # Handle JSON (text input)
    else:
        data = request.get_json()
        user_message = data.get('message', '')
        if user_message:
            llm_result = groq_llm(user_message)
            return jsonify({'response': llm_result})
        return jsonify({'response': 'No message provided.'})


if __name__ == '__main__':
    app.run(debug=True)
