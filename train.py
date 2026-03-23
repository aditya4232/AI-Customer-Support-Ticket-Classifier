import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Step 3: Load & Prepare Data
df = pd.read_csv("telecom_cc.csv")
df.columns = ["text", "category", "priority"]
print(df.head())

# Step 4: Clean Text
def clean_text(text):
   text = str(text).lower()
   text = re.sub(r'[^a-z\s]', '', text)
   return text

df["text"] = df["text"].apply(clean_text)

# Step 5: Convert Text → Numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Step 6: Train Model (Category)
model_category = MultinomialNB()
model_category.fit(X, df["category"])

# Step 7: Train Model (Priority)
model_priority = MultinomialNB()
model_priority.fit(X, df["priority"])

# Step 8: Test Prediction
def predict_ticket(text):
   text = clean_text(text)
   X_test = vectorizer.transform([text])

   category = model_category.predict(X_test)[0]
   priority = model_priority.predict(X_test)[0]

   return category, priority

print(predict_ticket("my internet is not working"))

# Step 9: Save Model
with open("model.pkl", "wb") as f:
   pickle.dump((vectorizer, model_category, model_priority), f)
