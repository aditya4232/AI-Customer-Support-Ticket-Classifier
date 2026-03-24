"""
train_advanced.py
=================
Trains 3 models on database-tele-data(telecom_cc).csv:
  1. TF-IDF + Naive Bayes  (baseline, saved as model.pkl)
  2. Neural Network – Embedding + Dense  (saved as nn_model/)
  3. LSTM  (saved as lstm_model/)
Also saves:
  - tokenizer.pkl  (shared Keras tokenizer for NN/LSTM)
  - label_encoders.pkl  (LabelEncoders for category + priority)
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

# ─── sklearn ────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── TensorFlow / Keras ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, LSTM as KerasLSTM,
    GlobalAveragePooling1D, Bidirectional, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_FILE    = "database-tele-data(telecom_cc).csv"
MAX_WORDS    = 10_000
MAX_LEN      = 100
EMBED_DIM    = 64
BATCH_SIZE   = 32
EPOCHS       = 25


# ============================================================
# 1. Load & clean
# ============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\n📂  Loading dataset …")
df = pd.read_csv(DATA_FILE)
# Expected columns: ticket_text, category, priority  (and optionally reason)
df = df.rename(columns={df.columns[0]: "text",
                         df.columns[1]: "category",
                         df.columns[2]: "priority"})
df = df.dropna(subset=["text", "category", "priority"])
df["text"] = df["text"].apply(clean_text)
print(f"   Rows after cleaning : {len(df)}")
print(f"   Categories ({df['category'].nunique()}) : {df['category'].unique()}")
print(f"   Priorities  ({df['priority'].nunique()}) : {df['priority'].unique()}")


# ============================================================
# 2. Label encoding
# ============================================================
le_cat  = LabelEncoder()
le_pri  = LabelEncoder()
df["cat_enc"] = le_cat.fit_transform(df["category"])
df["pri_enc"] = le_pri.fit_transform(df["priority"])

with open("label_encoders.pkl", "wb") as f:
    pickle.dump((le_cat, le_pri), f)
print("\n✅  label_encoders.pkl saved.")


# ============================================================
# 3. Train / test split
# ============================================================
X_train_txt, X_test_txt, y_cat_tr, y_cat_te, y_pri_tr, y_pri_te = train_test_split(
    df["text"].values,
    df["cat_enc"].values,
    df["pri_enc"].values,
    test_size=0.15, random_state=42, stratify=df["cat_enc"]
)


# ============================================================
# 4. MODEL A  –  TF-IDF + Multinomial Naive Bayes
# ============================================================
print("\n" + "="*55)
print("MODEL A  ─  TF-IDF + Naive Bayes")
print("="*55)

tfidf = TfidfVectorizer(max_features=MAX_WORDS, ngram_range=(1, 2),
                        sublinear_tf=True)
X_tfidf_tr = tfidf.fit_transform(X_train_txt)
X_tfidf_te = tfidf.transform(X_test_txt)

nb_cat = MultinomialNB(alpha=0.5)
nb_pri = MultinomialNB(alpha=0.5)
nb_cat.fit(X_tfidf_tr, y_cat_tr)
nb_pri.fit(X_tfidf_tr, y_pri_tr)

cat_preds_nb = nb_cat.predict(X_tfidf_te)
pri_preds_nb = nb_pri.predict(X_tfidf_te)

print("\n─ Category report (NB):")
print(classification_report(y_cat_te, cat_preds_nb,
                             target_names=le_cat.classes_))
print("─ Priority report (NB):")
print(classification_report(y_pri_te, pri_preds_nb,
                             target_names=le_pri.classes_))

with open("model.pkl", "wb") as f:
    pickle.dump((tfidf, nb_cat, nb_pri), f)
print("✅  model.pkl saved  (TF-IDF + NB).")


# ============================================================
# 5. Shared Keras Tokenizer (for NN + LSTM)
# ============================================================
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_txt)

X_seq_tr = pad_sequences(tokenizer.texts_to_sequences(X_train_txt),
                          maxlen=MAX_LEN, padding="post", truncating="post")
X_seq_te = pad_sequences(tokenizer.texts_to_sequences(X_test_txt),
                          maxlen=MAX_LEN, padding="post", truncating="post")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅  tokenizer.pkl saved.")

NUM_CATS = df["category"].nunique()
NUM_PRIS = df["priority"].nunique()

# One-hot labels for Keras
y_cat_oh_tr = tf.keras.utils.to_categorical(y_cat_tr, NUM_CATS)
y_cat_oh_te = tf.keras.utils.to_categorical(y_cat_te, NUM_CATS)
y_pri_oh_tr = tf.keras.utils.to_categorical(y_pri_tr, NUM_PRIS)
y_pri_oh_te = tf.keras.utils.to_categorical(y_pri_te, NUM_PRIS)

# Early stopping shared
early_stop = EarlyStopping(monitor="val_loss", patience=4,
                            restore_best_weights=True)
lr_reduce  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, min_lr=1e-5, verbose=0)


# ============================================================
# 6. MODEL B  –  Embedding + Dense (Neural Network)
# ============================================================
print("\n" + "="*55)
print("MODEL B  ─  Embedding + Dense (Neural Network)")
print("="*55)


def build_dense_model(num_classes: int) -> Sequential:
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax"),
    ], name=f"dense_{num_classes}")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ─ Category NN
nn_cat = build_dense_model(NUM_CATS)
nn_cat.fit(X_seq_tr, y_cat_oh_tr,
           epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15,
           callbacks=[early_stop, lr_reduce], verbose=1)

# ─ Priority NN
nn_pri = build_dense_model(NUM_PRIS)
nn_pri.fit(X_seq_tr, y_pri_oh_tr,
           epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15,
           callbacks=[early_stop, lr_reduce], verbose=1)

cat_preds_nn = np.argmax(nn_cat.predict(X_seq_te), axis=1)
pri_preds_nn = np.argmax(nn_pri.predict(X_seq_te), axis=1)

print("\n─ Category report (NN):")
print(classification_report(y_cat_te, cat_preds_nn,
                             target_names=le_cat.classes_))
print("─ Priority report (NN):")
print(classification_report(y_pri_te, pri_preds_nn,
                             target_names=le_pri.classes_))

os.makedirs("nn_model", exist_ok=True)
nn_cat.save("nn_model/category_model.h5")
nn_pri.save("nn_model/priority_model.h5")
print("✅  nn_model/ saved  (Embedding + Dense).")


# ============================================================
# 7. MODEL C  –  Bidirectional LSTM
# ============================================================
print("\n" + "="*55)
print("MODEL C  ─  Bidirectional LSTM")
print("="*55)


def build_lstm_model(num_classes: int) -> Sequential:
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(KerasLSTM(64, return_sequences=True, dropout=0.2,
                                 recurrent_dropout=0.2)),
        Bidirectional(KerasLSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name=f"lstm_{num_classes}")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ─ Category LSTM
lstm_cat = build_lstm_model(NUM_CATS)
lstm_cat.fit(X_seq_tr, y_cat_oh_tr,
             epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15,
             callbacks=[early_stop, lr_reduce], verbose=1)

# ─ Priority LSTM
lstm_pri = build_lstm_model(NUM_PRIS)
lstm_pri.fit(X_seq_tr, y_pri_oh_tr,
             epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15,
             callbacks=[early_stop, lr_reduce], verbose=1)

cat_preds_lstm = np.argmax(lstm_cat.predict(X_seq_te), axis=1)
pri_preds_lstm = np.argmax(lstm_pri.predict(X_seq_te), axis=1)

print("\n─ Category report (LSTM):")
print(classification_report(y_cat_te, cat_preds_lstm,
                             target_names=le_cat.classes_))
print("─ Priority report (LSTM):")
print(classification_report(y_pri_te, pri_preds_lstm,
                             target_names=le_pri.classes_))

os.makedirs("lstm_model", exist_ok=True)
lstm_cat.save("lstm_model/category_model.h5")
lstm_pri.save("lstm_model/priority_model.h5")
print("✅  lstm_model/ saved  (Bidirectional LSTM).")

print("\n🎉  All models trained and saved successfully!")
print("   model.pkl, tokenizer.pkl, label_encoders.pkl")
print("   nn_model/category_model.h5  nn_model/priority_model.h5")
print("   lstm_model/category_model.h5  lstm_model/priority_model.h5")
