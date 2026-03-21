# AI Customer Support Ticket Classifier - Architecture Plan

## Project Overview
Build an AI-powered customer support ticket classification system that automatically categorizes incoming tickets, assigns priority levels, and provides reasoning for classifications. The system will use classical machine learning (scikit-learn with TF-IDF) for efficient and interpretable classification.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Support Ticket                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Text Preprocessing Module                          │
│  - Lowercase conversion                                         │
│  - Punctuation removal                                          │
│  - Stopword removal                                              │
│  - Text normalization                                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              TF-IDF Vectorization                                │
│  - Word-level TF-IDF                                            │
│  - Character n-grams (for robustness)                           │
│  - Feature selection                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│  Category Classifier      │   │  Priority Classifier      │
│  (Multinomial NB / LR)    │   │  (Random Forest / LR)     │
│  Output: Category + Prob  │   │  Output: Priority         │
└───────────────────────────┘   └───────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Classification Output                              │
│  - Category (Sales, Technical, Billing, etc.)                  │
│  - Priority (High, Medium, Low)                                │
│  - Confidence Score                                             │
│  - Reasoning (key phrases extracted)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Training Data Module (`data/`)
- **training_data.py**: 100+ labeled ticket examples across categories
- **categories.json**: Define all possible categories and priorities

### 2. Preprocessing Module (`src/preprocessing/`)
- **text_cleaner.py**: Text cleaning and normalization
- **tokenizer.py**: Custom tokenization strategies

### 3. Feature Extraction (`src/features/`)
- **tfidf_vectorizer.py**: TF-IDF vectorization with optimized parameters

### 4. Models (`src/models/`)
- **category_classifier.py**: Category prediction with probability
- **priority_classifier.py**: Priority level prediction

### 5. Classifier System (`src/classifier/`)
- **ticket_classifier.py**: Main classifier combining both models
- **reasoning.py**: Extract key phrases for justification

### 6. Inference (`src/api/`)
- **predict.py**: Prediction interface for new tickets

## Data Categories

Based on the examples, we'll support:
- **Sales / Plan Upgrade** - Customer inquiries about plans, upgrades, pricing
- **Technical Support** - Technical issues, bugs, how-to questions
- **Billing / Payments** - Payment issues, invoices, subscriptions
- **Account** - Login, password, account settings
- **General Inquiry** - Other questions

## Priority Levels
- **High**: Urgent issues, system outages, critical bugs
- **Medium**: Standard issues requiring attention
- **Low**: General inquiries, information requests

## Implementation Steps

### Phase 1: Data & Preprocessing
1. Create 100 labeled training examples
2. Build preprocessing pipeline

### Phase 2: Model Training
3. Train TF-IDF vectorizer
4. Train category classifier with probability
5. Train priority classifier

### Phase 3: System Integration
6. Create unified classifier
7. Add reasoning extraction

### Phase 4: Testing & Validation
8. Test with example tickets
9. Validate accuracy metrics

## Example Input/Output

**Input:**
```
I would like to upgrade my current plan to the premium package.
Please let me know the steps and the price difference.
```

**Expected Output:**
```
Category: Sales / Plan Upgrade (95% confidence)
Priority: Low
Reason: Customer requesting product upgrade information
```

## Technology Stack
- **Python 3.x**
- **scikit-learn**: TF-IDF, classifiers (Naive Bayes, Logistic Regression)
- **pandas**: Data handling
- **numpy**: Numerical operations

## Files to Create
```
/data/
  - training_data.py
  - categories.json
/src/
  - __init__.py
  - preprocessing/
    - __init__.py
    - text_cleaner.py
  - features/
    - __init__.py
    - tfidf_vectorizer.py
  - models/
    - __init__.py
    - category_classifier.py
    - priority_classifier.py
  - classifier/
    - __init__.py
    - ticket_classifier.py
    - reasoning.py
  - api/
    - __init__.py
    - predict.py
/tests/
  - test_classifier.py
main.py
requirements.txt
README.md
```
