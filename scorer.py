import logging
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Try to import spaCy for better text processing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
    logging.info("Using spaCy for enhanced text processing in scorer")
except:
    HAS_SPACY = False
    logging.warning("spaCy not available for enhanced scoring - falling back to basic processing")

def score_resume_against_job(resume_data, job_description):
    """
    Score a resume against a job description
    
    Args:
        resume_data (dict): Parsed resume data
        job_description (str): Job description text
    
    Returns:
        tuple: (score, details)
            - score: int (0-100)
            - details: dict containing detailed scoring information
    """
    logging.debug("Scoring resume against job description")
    
    # Extract skills list from resume
    resume_skills = resume_data.get('skills', [])
    
    # Extract experience from resume
    resume_experience = resume_data.get('experience', [])
    
    # Extract other info
    resume_name = resume_data.get('name', '')
    resume_email = resume_data.get('email', '')
    
    # Combine all resume data into a single text for comprehensive analysis
    resume_text = " ".join([
        resume_name,
        resume_email,
        " ".join(resume_skills),
        " ".join(resume_experience)
    ])
    
    # Clean and normalize text
    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)
    
    # Calculate skill match scores
    skills_match = calculate_skills_match(resume_skills, job_description)
    
    # Calculate content match with TF-IDF and NLP
    content_match_score = calculate_content_match(resume_text, job_description)
    
    # No longer extracting key terms since this feature was removed
    
    # Calculate skill coverage percentage
    skill_coverage = len(skills_match) / max(len(resume_skills), 1) * 100 if resume_skills else 0
    
    # Calculate experience relevance
    experience_text = " ".join(resume_experience)
    exp_relevance = 0
    if experience_text and job_description:
        # Simple TF-IDF vectorization for experience relevance
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([experience_text, job_description])
            exp_relevance = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        except:
            exp_relevance = 0
    
    # Calculate detailed scores
    skills_score = sum(skills_match.values()) / max(len(skills_match), 1) * 100 if skills_match else 0
    
    # Create detailed breakdown for the report
    score_components = {
        "skills_score": round(skills_score, 1),
        "content_match": round(content_match_score, 1),
        "skill_coverage": round(skill_coverage, 1),
        "experience_relevance": round(exp_relevance, 1)
    }
    
    # Calculate final score (weighted average with adjusted weights)
    # 50% for skills match, 30% for content match, 20% for experience relevance
    final_score = int((0.5 * skills_score) + 
                      (0.3 * content_match_score) + 
                      (0.2 * exp_relevance))
    
    # Ensure score is between 0 and 100
    final_score = max(0, min(100, final_score))
    
    # Create detailed result dictionary - simplified without key terms
    details = {
        "score": final_score,
        "matched_skills": skills_match
    }
    
    # Add score category label
    if final_score >= 85:
        details["category"] = "Excellent Match"
    elif final_score >= 70:
        details["category"] = "Strong Match"
    elif final_score >= 50:
        details["category"] = "Moderate Match"
    elif final_score >= 30:
        details["category"] = "Weak Match"
    else:
        details["category"] = "Poor Match"
    
    return final_score, details

def clean_text(text):
    """Clean and normalize text for better comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_skills_match(resume_skills, job_description):
    """
    Calculate how well the resume skills match with job description
    
    Returns:
        dict: Dictionary of skills with match scores (0.0-1.0)
    """
    skills_match = {}
    job_desc_lower = job_description.lower()
    
    # Enhanced skill detection with NLP if available
    job_entities = []
    job_keywords = []
    
    if HAS_SPACY:
        try:
            # Process job description with spaCy for better context
            job_doc = nlp(job_description)
            
            # Extract key entities that might be related to skills
            job_entities = [
                ent.text.lower() for ent in job_doc.ents 
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "GPE"]
            ]
            
            # Extract noun chunks as potential multi-word skills
            job_keywords = [chunk.text.lower() for chunk in job_doc.noun_chunks]
            
            logging.debug(f"Extracted {len(job_entities)} entities and {len(job_keywords)} keywords from job description")
        except Exception as e:
            logging.warning(f"Error using spaCy for skill matching: {str(e)}")
    
    for skill in resume_skills:
        skill_lower = skill.lower()
        
        # Initialize scores
        exact_match_score = 0
        proximity_score = 0
        context_score = 0
        
        # 1. Check for exact matches in the job description
        if skill_lower in job_desc_lower:
            # Calculate importance score based on frequency and position
            # Position: Earlier mentions might be more important
            position_score = 1.0
            first_pos = job_desc_lower.find(skill_lower)
            if first_pos > -1:
                # Normalize position (earlier is better)
                position_score = 1.0 - (first_pos / len(job_desc_lower))
            
            # Frequency: More mentions might indicate importance
            frequency = job_desc_lower.count(skill_lower)
            frequency_score = min(frequency / 3, 1.0)  # Cap at 1.0
            
            exact_match_score = (0.7 * frequency_score + 0.3 * position_score)
        
        # 2. Check for word proximity matches (for multi-word skills)
        if ' ' in skill_lower and not exact_match_score:
            skill_words = skill_lower.split()
            if all(word in job_desc_lower for word in skill_words):
                # Calculate average distance between words
                word_positions = [job_desc_lower.find(word) for word in skill_words]
                avg_distance = sum([abs(word_positions[i] - word_positions[i-1]) 
                                  for i in range(1, len(word_positions))]) / max(1, len(word_positions) - 1)
                
                # Closer words get higher scores (normalized)
                proximity_score = 1.0 / (1.0 + (avg_distance / 100))
        
        # 3. Add context-based matching with NLP entities (if available)
        if HAS_SPACY and not (exact_match_score or proximity_score):
            # Check if skill is part of extracted entities or keywords
            if any(skill_lower in entity for entity in job_entities):
                context_score = 0.8  # High score for entity match
            elif any(skill_lower in keyword for keyword in job_keywords):
                context_score = 0.6  # Medium score for keyword match
            # Check for partial matches in longer terms
            elif any(skill_lower in entity and len(entity) > len(skill_lower) + 3 
                    for entity in job_entities + job_keywords):
                context_score = 0.4  # Lower score for partial match
        
        # Combine all scores and take the maximum
        combined_score = max(exact_match_score, proximity_score, context_score)
        
        # Only include skills with non-zero scores
        if combined_score > 0:
            skills_match[skill] = round(combined_score, 2)
    
    return skills_match

def calculate_content_match(resume_text, job_description):
    """
    Calculate similarity between resume text and job description using TF-IDF and cosine similarity
    
    Returns:
        float: Similarity score (0-100)
    """
    # Process with spaCy for NLP enhancements if available
    resume_keywords = []
    job_keywords = []
    
    if HAS_SPACY:
        try:
            # Process with spaCy
            resume_doc = nlp(resume_text)
            job_doc = nlp(job_description)
            
            # Extract entities and noun chunks
            resume_entities = [ent.text.lower() for ent in resume_doc.ents]
            job_entities = [ent.text.lower() for ent in job_doc.ents]
            
            # Get noun chunks (multi-word phrases that could be important)
            resume_keywords = [chunk.text.lower() for chunk in resume_doc.noun_chunks]
            job_keywords = [chunk.text.lower() for chunk in job_doc.noun_chunks]
            
            # Add these to the texts for improved matching
            resume_text += " " + " ".join(resume_entities + resume_keywords)
            job_description += " " + " ".join(job_entities + job_keywords)
            
            logging.debug("Enhanced content matching with NLP-extracted entities and keywords")
        except Exception as e:
            logging.warning(f"Error using spaCy for content matching: {str(e)}")
    
    # Create TF-IDF vectorizer with enhanced params
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        max_features=5000,   # Limit features for efficiency
        min_df=1,            # Minimum document frequency
        max_df=0.9           # Maximum document frequency
    )
    
    try:
        # Vectorize the texts
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage
        similarity_score = cosine_sim * 100
        
        # If we're using spaCy, also compute a key-term match score
        additional_score = 0
        if HAS_SPACY and resume_keywords and job_keywords:
            # Count keyword matches 
            common_keywords = set(resume_keywords).intersection(set(job_keywords))
            if common_keywords:
                # Compute match percentage with a max cap
                keyword_match = min(len(common_keywords) / max(len(job_keywords), 1) * 100, 100)
                # Weight this less than the cosine similarity
                additional_score = keyword_match * 0.3
        
        # Combine the scores
        weighted_score = (similarity_score * 0.7) + additional_score
        
        # Apply a moderate curve to avoid overly high scores
        final_score = min(weighted_score * 1.1, 100) 
        
        return final_score
    except Exception as e:
        logging.error(f"Error calculating content match: {str(e)}")
        return 0
        
def extract_key_terms(text):
    """Extract important terms from text using frequency analysis"""
    # This is a helper function for advanced text analysis
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter common words and short words
    stop_words = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'with', 'on', 'at', 'from']
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Return most common words
    return [word for word, count in word_counts.most_common(20)]