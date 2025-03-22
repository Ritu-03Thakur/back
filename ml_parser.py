import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
import spacy
import os

class MLResumeParser:
    def __init__(self):
        self.skills_classifier = None
        self.model_path = os.path.join('models', 'skills_classifier.joblib')
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except:
            self.has_spacy = False
            logging.warning("spaCy model not available for ML parser")
    
    def train_skills_classifier(self, training_data):
        """
        Train a classifier to identify skills in text
        
        Args:
            training_data: List of (text, is_skill) pairs
        """
        logging.info(f"Training skills classifier with {len(training_data)} examples")
        
        # Extract features and labels
        X = []
        y = []
        
        for text, is_skill in training_data:
            X.append(self._extract_features(text))
            y.append(1 if is_skill else 0)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        
        # Set model
        self.skills_classifier = model
        
        # Return model metrics
        return {
            'accuracy': model.score(X, y),
            'feature_importance': dict(zip(self._get_feature_names(), model.feature_importances_))
        }
    
    def load_skills_classifier(self):
        """Load trained skills classifier if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.skills_classifier = joblib.load(self.model_path)
                logging.info("Loaded skills classifier model")
                return True
            else:
                logging.warning(f"Skills classifier model not found at {self.model_path}")
                return False
        except Exception as e:
            logging.error(f"Error loading skills classifier: {str(e)}")
            return False
    
    def is_skill(self, text):
        """Predict if a text is a skill"""
        if not self.skills_classifier:
            logging.warning("Skills classifier not loaded")
            return True  # Default to True to avoid missing skills
            
        features = self._extract_features(text)
        prediction = self.skills_classifier.predict([features])[0]
        
        return bool(prediction)
    
    def _extract_features(self, text):
        """Extract features from text for classification"""
        # Clean text
        text = text.strip()
        
        # Initialize features
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_digits': any(c.isdigit() for c in text),
            'capitalized': text[0].isupper() if text else False,
            'all_caps': text.isupper(),
            'contains_slash': '/' in text,
            'contains_plus': '+' in text,
            'contains_hyphen': '-' in text or '–' in text,
            'contains_period': '.' in text,
            'starts_with_verb': False,
            'has_technical_term': False,
        }
        
        # Check for technical terminology
        tech_terms = ['api', 'ui', 'ux', 'css', 'html', 'js', 'db', 'sql', 'nosql', 
                     'aws', 'ml', 'ai', 'ci', 'cd', 'qa', 'oop', 'mvc', 'rest']
        features['has_technical_term'] = any(term in text.lower() for term in tech_terms)
        
        # Use spaCy for linguistic features if available
        if self.has_spacy:
            try:
                doc = self.nlp(text)
                
                # Check if starts with verb
                if len(doc) > 0:
                    features['starts_with_verb'] = doc[0].pos_ == "VERB"
                
                # Add entity type if detected
                if len(doc.ents) > 0:
                    features['is_entity'] = 1
                    features['is_org_entity'] = 1 if doc.ents[0].label_ == "ORG" else 0
                    features['is_product_entity'] = 1 if doc.ents[0].label_ == "PRODUCT" else 0
                else:
                    features['is_entity'] = 0
                    features['is_org_entity'] = 0
                    features['is_product_entity'] = 0
            except:
                # Default values if spaCy processing fails
                features['starts_with_verb'] = False
                features['is_entity'] = 0
                features['is_org_entity'] = 0
                features['is_product_entity'] = 0
        
        # Convert to flat list in consistent order
        return [features[f] for f in self._get_feature_names()]
    
    def _get_feature_names(self):
        """Get list of feature names in consistent order"""
        return [
            'length', 'word_count', 'has_digits', 'capitalized', 'all_caps',
            'contains_slash', 'contains_plus', 'contains_hyphen', 'contains_period',
            'starts_with_verb', 'has_technical_term', 'is_entity', 'is_org_entity',
            'is_product_entity'
        ]