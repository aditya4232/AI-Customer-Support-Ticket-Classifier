"""
Reasoning component for extracting key phrases and justification.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter


class ReasoningExtractor:
    """Extract reasoning/key phrases from ticket text."""
    
    def __init__(self):
        """Initialize the reasoning extractor."""
        # Define keyword patterns for each category
        self.category_keywords = {
            'Sales / Plan Upgrade': [
                'upgrade', 'premium', 'plan', 'pricing', 'discount', 'subscription',
                'purchase', 'buy', 'cost', 'price', 'package', 'tier', 'enterprise',
                'business', 'features', 'trial', 'team', 'licenses', 'credits',
                'partner', 'invoice', 'demo', 'quote', 'refund'
            ],
            'Technical Support': [
                'error', 'bug', 'crash', 'broken', 'not working', 'failed', 'issue',
                'problem', 'help', 'how to', 'setup', 'configure', 'install',
                'integration', 'api', 'sync', 'upload', 'download', 'export',
                'login', 'password', 'authentication', 'permission', 'access',
                'performance', 'slow', 'freezing', 'timeout', 'connection'
            ],
            'Billing / Payments': [
                'charge', 'billing', 'payment', 'invoice', 'refund', 'subscription',
                'card', 'credit', 'debit', 'paypal', 'overdue', 'promo', 'code',
                'discount', 'vat', 'tax', 'currency', 'receipt', 'transaction'
            ],
            'Account': [
                'account', 'login', 'password', 'email', 'profile', 'username',
                'delete', 'close', 'reactivate', 'transfer', 'ownership', 'merge',
                'permissions', 'access', 'two-factor', 'authentication', 'verify',
                'settings', 'preferences', 'timezone', 'notification'
            ],
            'General Inquiry': [
                'what', 'how', 'where', 'when', 'do you', 'can i', 'is there',
                'information', 'question', 'help', 'learn', 'find', 'manual',
                'document', 'policy', 'terms', 'compliance', 'uptime'
            ]
        }
        
        # Priority indicators
        self.priority_indicators = {
            'High': [
                'urgent', 'critical', 'emergency', 'asap', 'immediately',
                'not working', 'system down', 'outage', 'security', 'breach',
                'hacked', 'fraud', 'charged twice', 'account locked', 'crash',
                'broken', 'severe', 'major', 'production', 'downtime'
            ],
            'Medium': [
                'need help', 'issue', 'problem', 'error', 'not correct',
                'incorrect', 'failed', 'doesnt work', 'cant access',
                'configure', 'setup', 'integration', 'migrate'
            ],
            'Low': [
                'question', 'information', 'how do i', 'where can i', 'what is',
                'can i get', 'do you have', 'looking for', 'interested in',
                'wondering', 'curious', 'explain'
            ]
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Get all unique keywords
        all_keywords = []
        for keywords in self.category_keywords.values():
            all_keywords.extend(keywords)
        
        found_keywords = [w for w in words if w in all_keywords]
        
        return list(set(found_keywords))
    
    def extract_category_reasoning(self, text: str, predicted_category: str) -> str:
        """Generate reasoning for category prediction."""
        keywords = self.extract_keywords(text)
        
        # Get category-specific keywords
        category_keywords = self.category_keywords.get(predicted_category, [])
        matched_keywords = [k for k in keywords if k in category_keywords]
        
        text_lower = text.lower()

        # Generate reasoning based on matched keywords and high-signal phrases
        if 'upgrade' in matched_keywords or 'premium' in matched_keywords or 'plan' in matched_keywords:
            return "Customer requesting product upgrade information"
        elif 'error' in matched_keywords or 'bug' in matched_keywords or 'crash' in matched_keywords:
            return "Customer reporting technical issue or error"
        elif 'charge' in matched_keywords or 'payment' in matched_keywords or 'billing' in matched_keywords:
            return "Customer inquiry about billing or payment"
        elif 'account' in matched_keywords or 'login' in matched_keywords or 'password' in matched_keywords:
            return "Customer request related to account management"
        elif 'how' in matched_keywords or 'what' in matched_keywords or 'where' in matched_keywords:
            return "Customer asking general information question"
        else:
            return f"Keywords detected: {', '.join(matched_keywords[:5])}"
    
    def extract_priority_reasoning(self, text: str, predicted_priority: str) -> str:
        """Generate reasoning for priority prediction."""
        text_lower = text.lower()
        
        # Check for high priority indicators
        high_count = sum(1 for indicator in self.priority_indicators['High'] 
                        if indicator in text_lower)
        medium_count = sum(1 for indicator in self.priority_indicators['Medium'] 
                          if indicator in text_lower)
        low_count = sum(1 for indicator in self.priority_indicators['Low'] 
                       if indicator in text_lower)
        
        # Generate reasoning
        if predicted_priority == 'High':
            if 'urgent' in text_lower or 'critical' in text_lower:
                return "Urgent or critical issue reported"
            elif 'not working' in text_lower or 'failed' in text_lower:
                return "System functionality issue affecting work"
            elif 'security' in text_lower or 'hacked' in text_lower:
                return "Security-related concern"
            else:
                return "Issue requiring immediate attention"
        elif predicted_priority == 'Medium':
            return "Standard issue requiring support attention"
        else:
            return "General inquiry or information request"
    
    def generate_full_reasoning(self, text: str, category: str, priority: str) -> Dict[str, str]:
        """Generate complete reasoning for classification."""
        return {
            'category_reasoning': self.extract_category_reasoning(text, category),
            'priority_reasoning': self.extract_priority_reasoning(text, priority),
            'extracted_keywords': self.extract_keywords(text)
        }
    
    def explain_prediction(self, text: str, category: str, category_confidence: float,
                          priority: str, priority_confidence: float) -> Dict:
        """Generate detailed explanation of prediction."""
        reasoning = self.generate_full_reasoning(text, category, priority)
        
        return {
            'input_text': text,
            'category': {
                'predicted': category,
                'confidence': category_confidence,
                'reasoning': reasoning['category_reasoning']
            },
            'priority': {
                'predicted': priority,
                'confidence': priority_confidence,
                'reasoning': reasoning['priority_reasoning']
            },
            'extracted_keywords': reasoning['extracted_keywords']
        }


def extract_reasoning(text: str, category: str, priority: str) -> Dict[str, str]:
    """Convenience function to extract reasoning."""
    extractor = ReasoningExtractor()
    return extractor.generate_full_reasoning(text, category, priority)


if __name__ == "__main__":
    # Test the reasoning extractor
    extractor = ReasoningExtractor()
    
    test_texts = [
        "I would like to upgrade my current plan to the premium package.",
        "I'm getting an error when trying to login",
        "I was charged twice for my subscription"
    ]
    
    print("Reasoning Extraction Test:")
    print("-" * 50)
    for text in test_texts:
        keywords = extractor.extract_keywords(text)
        print(f"Text: {text}")
        print(f"Keywords: {keywords}")
        print()
