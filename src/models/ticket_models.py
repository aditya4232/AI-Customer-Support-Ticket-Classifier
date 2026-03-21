"""
Classifier models for the AI Customer Support Ticket Classifier.
Supports TF-IDF based classification with Industry, Category, and Priority.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


class IndustryClassifier:
    """Industry classifier for support tickets."""
    
    def __init__(self):
        """Initialize the industry classifier."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        
        self.industry_encoder = LabelEncoder()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train the industry classifier."""
        y_encoded = self.industry_encoder.fit_transform(y_train)
        
        X_transformed = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_transformed, y_encoded)
        
        train_pred = self.model.predict(X_transformed)
        accuracy = np.mean(train_pred == y_encoded)
        
        return {'train_accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict industries for input text."""
        X_transformed = self.vectorizer.transform(X)
        
        predictions = self.model.predict(X_transformed)
        probabilities = self.model.predict_proba(X_transformed)
        
        industries = self.industry_encoder.inverse_transform(predictions)
        
        return industries, probabilities
    
    def predict_industry(self, text: str) -> Dict[str, Any]:
        """Predict industry for a single text."""
        X = np.array([text])
        industries, probabilities = self.predict(X)
        
        max_prob = np.max(probabilities[0])
        
        return {
            'industry': industries[0],
            'confidence': float(max_prob),
            'all_probabilities': {
                ind: float(prob) 
                for ind, prob in zip(self.industry_encoder.classes_, probabilities[0])
            }
        }


class CategoryClassifier:
    """Category classifier using TF-IDF."""
    
    def __init__(self, use_transformer: bool = False):
        """Initialize the category classifier."""
        self.use_transformer = use_transformer
        self.vectorizer = None
        self.category_encoder = None
        self.model = None
        self.pipeline = None
    
    def _create_pipeline(self) -> Pipeline:
        """Create the TF-IDF classification pipeline."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        
        pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        return pipeline
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train the category classifier."""
        self.category_encoder = LabelEncoder()
        
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        train_pred = self.pipeline.predict(X_train)
        accuracy = np.mean(train_pred == y_train)
        
        return {'train_accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict categories for input text."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        # Pipeline already returns string labels
        categories = predictions
        
        return categories, probabilities
    
    def predict_with_confidence(self, text: str) -> Dict[str, Any]:
        """Predict category for a single text with confidence score."""
        X = np.array([text])
        categories, probabilities = self.predict(X)
        
        max_prob = np.max(probabilities[0])
        predicted_category = categories[0]
        
        class_names = self.pipeline.classes_
        
        return {
            'category': predicted_category,
            'confidence': float(max_prob),
            'all_probabilities': {
                cat: float(prob) 
                for cat, prob in zip(class_names, probabilities[0])
            }
        }


class PriorityClassifier:
    """Priority classifier for support tickets."""
    
    def __init__(self):
        """Initialize the priority classifier."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.priority_encoder = LabelEncoder()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train the priority classifier."""
        y_encoded = self.priority_encoder.fit_transform(y_train)
        
        X_transformed = self.vectorizer.fit_transform(X_train)
        
        self.model.fit(X_transformed, y_encoded)
        
        train_pred = self.model.predict(X_transformed)
        accuracy = np.mean(train_pred == y_encoded)
        
        return {'train_accuracy': accuracy}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict priorities for input text."""
        X_transformed = self.vectorizer.transform(X)
        
        predictions = self.model.predict(X_transformed)
        probabilities = self.model.predict_proba(X_transformed)
        
        priorities = self.priority_encoder.inverse_transform(predictions)
        
        return priorities, probabilities
    
    def predict_priority(self, text: str) -> Dict[str, Any]:
        """Predict priority for a single text."""
        X = np.array([text])
        priorities, probabilities = self.predict(X)
        
        max_prob = np.max(probabilities[0])
        
        return {
            'priority': priorities[0],
            'confidence': float(max_prob),
            'all_probabilities': {
                pri: float(prob) 
                for pri, prob in zip(self.priority_encoder.classes_, probabilities[0])
            }
        }


class TicketClassifier:
    """Combined ticket classifier for industry, category, and priority."""
    
    def __init__(self, use_transformer: bool = False):
        """Initialize the ticket classifier."""
        self.industry_classifier = IndustryClassifier()
        self.category_classifier = CategoryClassifier(use_transformer=use_transformer)
        self.priority_classifier = PriorityClassifier()
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_industry_train: np.ndarray,
              y_category_train: np.ndarray, y_priority_train: np.ndarray) -> Dict[str, Dict]:
        """Train all classifiers."""
        print("Training industry classifier...")
        industry_metrics = self.industry_classifier.train(X_train, y_industry_train)
        
        print("Training category classifier...")
        category_metrics = self.category_classifier.train(X_train, y_category_train)
        
        print("Training priority classifier...")
        priority_metrics = self.priority_classifier.train(X_train, y_priority_train)
        
        self.is_trained = True
        
        return {
            'industry': industry_metrics,
            'category': category_metrics,
            'priority': priority_metrics
        }
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a support ticket."""
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")

        industry_result = self.industry_classifier.predict_industry(text)
        category_result = self.category_classifier.predict_with_confidence(text)
        priority_result = self.priority_classifier.predict_priority(text)

        text_lower = text.lower()
        duplicate_billing_detected = (
            any(term in text_lower for term in ['charged twice', 'double charge', 'duplicate charge', 'charged two times'])
            and any(term in text_lower for term in ['subscription', 'billing', 'payment', 'refund'])
        )

        if duplicate_billing_detected:
            category_result['category'] = 'Billing'
            category_result['confidence'] = max(float(category_result['confidence']), 0.95)
            category_result['all_probabilities']['Billing'] = max(
                float(category_result['all_probabilities'].get('Billing', 0.0)),
                0.95
            )

            if priority_result['priority'] == 'Low':
                priority_result['priority'] = 'Medium'
                priority_result['confidence'] = max(float(priority_result['confidence']), 0.75)
                priority_result['all_probabilities']['Medium'] = max(
                    float(priority_result['all_probabilities'].get('Medium', 0.0)),
                    0.75
                )

        return {
            'industry': industry_result['industry'],
            'industry_confidence': industry_result['confidence'],
            'industry_probabilities': industry_result['all_probabilities'],
            'category': category_result['category'],
            'category_confidence': category_result['confidence'],
            'category_probabilities': category_result['all_probabilities'],
            'priority': priority_result['priority'],
            'priority_confidence': priority_result['confidence'],
            'priority_probabilities': priority_result['all_probabilities']
        }
    
    def classify_batch(self, texts: np.ndarray) -> List[Dict[str, Any]]:
        """Classify multiple support tickets."""
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        industry_preds, industry_probs = self.industry_classifier.predict(texts)
        category_preds, category_probs = self.category_classifier.predict(texts)
        priority_preds, priority_probs = self.priority_classifier.predict(texts)
        
        results = []
        for i in range(len(texts)):
            results.append({
                'text': texts[i],
                'industry': industry_preds[i],
                'industry_confidence': float(np.max(industry_probs[i])),
                'category': category_preds[i],
                'category_confidence': float(np.max(category_probs[i])),
                'priority': priority_preds[i],
                'priority_confidence': float(np.max(priority_probs[i]))
            })
        
        return results
    
    def save(self, filepath: str):
        """Save the classifier to files."""
        joblib.dump(self.industry_classifier, f"{filepath}_industry.pkl")
        joblib.dump(self.category_classifier, f"{filepath}_category.pkl")
        joblib.dump(self.priority_classifier, f"{filepath}_priority.pkl")
    
    def load(self, filepath: str):
        """Load the classifier from files."""
        self.industry_classifier = joblib.load(f"{filepath}_industry.pkl")
        self.category_classifier = joblib.load(f"{filepath}_category.pkl")
        self.priority_classifier = joblib.load(f"{filepath}_priority.pkl")
        self.is_trained = True
