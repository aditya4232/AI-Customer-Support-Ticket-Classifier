"""
AI Customer Support Ticket Classifier
Main entry point for training and prediction.
"""

import sys
import os
import argparse
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import TicketDataLoader
from src.models.ticket_models import TicketClassifier
from src.classifier.reasoning import ReasoningExtractor


class TicketClassificationSystem:
    """Main system for ticket classification."""
    
    def __init__(self, use_transformer: bool = False):
        """
        Initialize the classification system.
        
        Args:
            use_transformer: Whether to use transformer-based model
        """
        self.classifier = TicketClassifier(use_transformer=use_transformer)
        self.reasoning_extractor = ReasoningExtractor()
        self.is_trained = False
    
    def train(self, data_path: str = "data/ticket_database.csv") -> dict:
        """
        Train the classification system.
        
        Args:
            data_path: Path to training data CSV
            
        Returns:
            Training metrics
        """
        print("Loading data...")
        loader = TicketDataLoader(data_path)
        loader.load_data()
        
        # Get all data for training
        X, y_industry, y_category, y_priority = loader.get_train_data()
        
        print(f"Total samples: {len(X)}")
        print(f"Industries: {np.unique(y_industry)}")
        print(f"Categories: {len(np.unique(y_category))} unique categories")
        
        # Train classifier on ALL data
        metrics = self.classifier.train(X, y_industry, y_category, y_priority)
        
        self.is_trained = True
        
        print(f"\nTraining complete!")
        
        return metrics
    
    def predict(self, ticket_text: str, include_reasoning: bool = True) -> dict:
        """
        Predict industry, category, and priority for a ticket.
        
        Args:
            ticket_text: Support ticket text
            include_reasoning: Whether to include reasoning
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        # Get classification
        classification = self.classifier.classify(ticket_text)
        
        if include_reasoning:
            reasoning = self.reasoning_extractor.generate_full_reasoning(
                ticket_text,
                classification['category'],
                classification['priority']
            )
            
            return {
                'industry': classification['industry'],
                'industry_confidence': classification['industry_confidence'],
                'category': classification['category'],
                'category_confidence': classification['category_confidence'],
                'category_probabilities': classification['category_probabilities'],
                'priority': classification['priority'],
                'priority_confidence': classification['priority_confidence'],
                'priority_probabilities': classification['priority_probabilities'],
                'reasoning': reasoning
            }
        
        return classification
    
    def predict_json(self, ticket_text: str) -> str:
        """
        Predict and return JSON output.
        
        Args:
            ticket_text: Support ticket text
            
        Returns:
            JSON string of prediction
        """
        result = self.predict(ticket_text)
        return json.dumps(result, indent=2)
    
    def save_model(self, filepath: str = "models/ticket_classifier"):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("System not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.classifier.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/ticket_classifier"):
        """Load a trained model."""
        self.classifier.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="AI Customer Support Ticket Classifier"
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the classifier'
    )
    parser.add_argument(
        '--predict',
        type=str,
        help='Predict category for given text'
    )
    parser.add_argument(
        '--use-transformer',
        action='store_true',
        help='Use transformer-based model'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/ticket_database.csv',
        help='Path to training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for prediction'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = TicketClassificationSystem(use_transformer=args.use_transformer)
    
    if args.train:
        # Train the system
        metrics = system.train(args.data_path)
        
        # Save model
        system.save_model()
        
        print("\nTraining Metrics:")
        print(json.dumps(metrics, indent=2))
    
    if args.predict:
        # Load model if available, otherwise train
        if not system.is_trained:
            try:
                system.load_model()
            except:
                print("No trained model found. Training first...")
                system.train(args.data_path)
        
        # Make prediction
        result = system.predict(args.predict)
        
        # Output result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    if not args.train and not args.predict:
        # Interactive mode
        print("=" * 70)
        print("   AI Customer Support Ticket Classifier (Multi-Industry)")
        print("=" * 70)
        
        # Train the system
        system.train(args.data_path)
        
        print("\n" + "=" * 70)
        print("Testing with example tickets:")
        print("=" * 70)
        
        # Test examples
        examples = [
            "I would like to upgrade my current plan to the premium package. Please let me know the steps and the price difference.",
            "The application keeps crashing when I try to export data",
            "I was charged twice for my subscription this month",
            "My phone isn't connecting to the network",
            "I need to schedule an appointment with Dr. Smith",
            "My online banking login isn't working",
            "I want to open a new savings account",
            "How do I track my order?",
            "The API integration is returning errors",
            "I suspect unauthorized activity on my account"
        ]
        
        for text in examples:
            print(f"\n{'='*70}")
            print(f"Input: {text}")
            result = system.predict(text)
            print(f"\n>>> Industry: {result['industry']} ({result['industry_confidence']:.1%})")
            print(f">>> Category: {result['category']} ({result['category_confidence']:.1%})")
            print(f">>> Priority: {result['priority']} ({result['priority_confidence']:.1%})")
            print(f">>> Reasoning: {result['reasoning']['category_reasoning']}")


if __name__ == "__main__":
    main()
