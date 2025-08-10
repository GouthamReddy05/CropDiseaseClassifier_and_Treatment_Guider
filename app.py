from flask import Flask, render_template, request, jsonify
import requests
import os
from python_files.main import predict_disease, process_image, encode_output
from python_files.rag import run_rag_pipeline

app = Flask(__name__)

# Correct Gemini endpoint
gemini_api_key = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def gemini_llm(prompt):
    """Send prompt to Gemini API and get the response."""
    headers = {"Content-Type": "application/json"}
    params = {"key": gemini_api_key}
    data = {
    "contents": [
        {"parts": [{"text": prompt}]}
    ],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 1024
    }
}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
        if response.ok:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # If the request is form-data (for file upload)
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        if 'image' in request.files:
            image_file = request.files['image']
            image_path = os.path.join('uploads', image_file.filename)
            os.makedirs('uploads', exist_ok=True)
            image_file.save(image_path)

            processed_image_path = process_image(image_path)
            prediction = predict_disease(processed_image_path)
            prediction_text = f"Predicted Disease: {prediction}"

            encoded_response = encode_output(prediction_text)
            rag_output = run_rag_pipeline(prediction_text)

            combined_prompt = (
                f"Here is the disease prediction: {prediction_text}\n\n"
                f"Additional information from knowledge base: {rag_output}\n\n"
                f"Based on the above, give a detailed explanation and treatment plan."
            )

            llm_result = gemini_llm(combined_prompt)

            os.remove(processed_image_path)

            return jsonify({
                'prediction': prediction_text,
                'knowledge': rag_output,
                'llm_response': llm_result
            })
        else:
            return jsonify({'response': 'No image uploaded.'})

    # If the request is JSON (for text input)
    else:
        data = request.get_json()
        user_message = data.get('message', '')
        if user_message:
            llm_result = gemini_llm(user_message)
            return jsonify({'response': llm_result})
        return jsonify({'response': 'No message provided.'})

if __name__ == '__main__':
    app.run(debug=True)
