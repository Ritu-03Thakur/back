import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

from utils.parser import extract_resume_data
from utils.scorer import score_resume_against_job

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'json'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload resume and compare against job description."""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    job_description = request.form.get('job_description', '')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        # Save file to temporary location
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        try:
            # Parse resume
            resume_data = extract_resume_data(file_path, file_extension)

            # Score against job description
            score = 0
            score_details = {}
            if job_description and resume_data:
                score, score_details = score_resume_against_job(resume_data, job_description)

            # Clean up temp file
            os.remove(file_path)

            # Return result as JSON
            return jsonify({
                "resume_data": resume_data,
                "score": score,
                "score_details": score_details,
                "job_description": job_description
            })

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            
            # Clean up even on failure
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({'error': f"Error processing file: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file type. Only PDF, DOCX, or JSON allowed.'}), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size errors."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
