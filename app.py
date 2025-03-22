import os
import logging
from flask import Flask, flash, request, redirect, render_template, url_for, session
from werkzeug.utils import secure_filename
import tempfile

from utils.parser import extract_resume_data
from utils.scorer import score_resume_against_job
# from utils.parser import parse_pdf, parse_docx, parse_json


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('index.html')


# calling parser.py for execution
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
        
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
            
            # Score against job description if provided
            score = 0
            score_details = {}
            if job_description and resume_data:
                score, score_details = score_resume_against_job(resume_data, job_description)
            
            # Store in session for result page
            session['resume_data'] = resume_data
            session['score'] = score
            session['score_details'] = score_details
            session['job_description'] = job_description
            
            # Clean up temp file
            os.remove(file_path)
            
            return redirect(url_for('result'))
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(request.url)
    else:
        flash(f'File type not allowed. Please upload PDF, DOCX, or JSON files.', 'danger')
        return redirect(request.url)

# calling scorer.py file for execution
@app.route('/result')
def result():
    # Get results from session
    resume_data = session.get('resume_data', {})
    score = session.get('score', 0)
    score_details = session.get('score_details', {})
    job_description = session.get('job_description', '')
    
    # Get matched skills
    skills_match = score_details.get('matched_skills', {}) if score_details else {}
    
    if not resume_data:
        flash('No resume data found. Please upload a resume first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('result.html', 
                          resume_data=resume_data, 
                          score=score, 
                          skills_match=skills_match,
                          job_description=job_description,
                          score_details=score_details)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 16MB.', 'danger')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An unexpected error occurred. Please try again.', 'danger')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)