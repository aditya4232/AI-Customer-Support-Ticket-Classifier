import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── Load primary database (490 rows) ─────────────────────────
df = pd.read_csv('database-tele-data(telecom_cc).csv')
df = df[['ticket_text', 'category', 'priority']].copy()
df = df.dropna(subset=['ticket_text', 'category', 'priority'])

print(f"Loaded {len(df)} rows  |  {df['category'].nunique()} categories  |  {df['priority'].nunique()} priorities")
print("Categories:", sorted(df['category'].unique()))
print("Priorities:", sorted(df['priority'].unique()))

# ── Clean text ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['ticket_text'].apply(clean_text)
# Combine ticket text + category label as feature for priority model
# (category is the strongest predictor of priority in this dataset)
df['pri_feature'] = df['text_clean'] + ' category ' + df['category'].str.lower()

# ── Train-test split (stratified) ────────────────────────────
X_train_cat, X_test_cat, y_cat_train, y_cat_test = train_test_split(
    df['text_clean'], df['category'],
    test_size=0.2, random_state=42, stratify=df['category'])

X_train_pri, X_test_pri, y_pri_train, y_pri_test = train_test_split(
    df['pri_feature'], df['priority'],
    test_size=0.2, random_state=42, stratify=df['priority'])

# ── Category pipeline ─────────────────────────────────────────
cat_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
    ('clf',   SGDClassifier(loss='modified_huber', random_state=42, max_iter=200)),
])
cat_pipeline.fit(X_train_cat, y_cat_train)
y_cat_pred = cat_pipeline.predict(X_test_cat)
print(f'\nCategory Accuracy : {accuracy_score(y_cat_test, y_cat_pred):.2f}')
print(classification_report(y_cat_test, y_cat_pred))

# ── Priority pipeline (text + true category label as feature) ─
pri_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1, 2))),
    ('clf',   SGDClassifier(loss='modified_huber', random_state=42, max_iter=200,
                            class_weight='balanced')),
])
pri_pipeline.fit(X_train_pri, y_pri_train)
y_pri_pred = pri_pipeline.predict(X_test_pri)
print(f'Priority Accuracy  : {accuracy_score(y_pri_test, y_pri_pred):.2f}')
print(classification_report(y_pri_test, y_pri_pred))

cv_scores = cross_val_score(pri_pipeline, df['pri_feature'], df['priority'],
                             cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f'Priority CV (5-fold): {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}')

# ── Save models ───────────────────────────────────────────────
joblib.dump(cat_pipeline, 'model.joblib')
joblib.dump(pri_pipeline, 'model_priority.joblib')
print('\nSaved: model.joblib  |  model_priority.joblib')
