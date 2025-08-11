from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback
from python_files.main import process_image, predict_disease, encode_output
from python_files.rag import run_rag_pipeline

load_dotenv()

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')  # Note: using uppercase for consistency

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def groq_llm(prompt, system_prompt=None):
    """Send prompt to Groq API and get the response."""
    if not groq_api_key:
        return "Error: Groq API key not configured. Please set GROQ_API_KEY in your .env file."
    
    try:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_api_key}"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )
        
        if not response.ok:
            return f"Error: {response.status_code} - {response.text}"
        
        result = response.json()
        
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "Error: Unexpected API response format."
    
    except Exception as e:
        print(f"Error communicating with Groq API: {e}")
        return f"Error communicating with Groq API: {str(e)}"

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle text-only chat messages."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Create a medical-focused system prompt
        system_prompt = """You are a helpful AI medical assistant. Provide informative, accurate medical information while always recommending users consult healthcare professionals for serious concerns. Be empathetic and helpful. If you're unsure about something, say so. Format your responses clearly with line breaks where appropriate."""
        
        llm_result = groq_llm(user_message, system_prompt)
        
        return jsonify({
            'reply': llm_result,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error occurred while processing your message.',
            'status': 'error'
        }), 500

@app.route('/process-image', methods=['POST'])
def process_image_route():
    """Handle image uploads - simplified version for basic image loading."""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Check if file is selected
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure the filename and save
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)
        
        # Get optional text message
        user_message = request.form.get('message', '').strip()
        
        print(f"✅ Image uploaded successfully: {image_path}")
        print(f"📁 File size: {os.path.getsize(image_path)} bytes")
        print(f"💬 User message: {user_message}")
        
        # For now, just return success with basic info
        # You can integrate your disease detection here
        
        response_message = f"""
🖼️ **Image Upload Successful!**

📝 **File Details:**
- Filename: {image_file.filename}
- Size: {round(os.path.getsize(image_path) / 1024, 2)} KB
- Path: {image_path}

💡 **Next Steps:**
- Image has been saved and is ready for processing
- You can now integrate your disease detection model
- The image path is available at: `{image_path}`

{f"**Your Message:** {user_message}" if user_message else ""}

**Note:** This is a basic image upload response. Integrate your disease detection logic in the `process_image_route()` function.
"""
        image = process_image(image_path)
        disease_prediction = predict_disease(image_path)
        encoded_output = encode_output(disease_prediction)  
        rag_output = run_rag_pipeline(disease_prediction)

        llm_response = groq_llm(
            f"Based on the disease prediction '{disease_prediction}', provide relevant information.",
            system_prompt="You are a helpful AI assistant providing information based on disease predictions."
        )
        
        # Clean up - remove uploaded file after showing it was processed
        try:
            # Comment out the cleanup for now so you can see the file was saved
            # os.remove(image_path)
            pass
        except Exception as cleanup_e:
            print(f"Cleanup error: {cleanup_e}")
        
        return jsonify({
            'reply': llm_response,
            'image_path': image_path,
            'filename': image_file.filename,
            'file_size': os.path.getsize(image_path),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"❌ Error in process-image endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'groq_api_configured': bool(groq_api_key),
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Maximum size allowed is 16MB.',
        'status': 'error'
    }), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors."""
    return jsonify({
        'error': 'Bad request. Please check your input.',
        'status': 'error'
    }), 400

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal server error. Please try again later.',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # Check API key
    if not groq_api_key:
        print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")
        print("Please set your Groq API key in the .env file.")
    else:
        print("✅ Groq API key configured")
    
    print(f"🚀 Starting Flask app...")
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"📝 Supported image formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📏 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    
    app.run(debug=True, host='0.0.0.0', port=5000)