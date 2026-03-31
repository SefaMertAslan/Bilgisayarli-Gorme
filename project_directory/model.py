import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Veriyi Yükle
df = pd.read_csv("veri.csv")

# 2. Metni Temizle
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[\w_]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["tweet_text"].apply(clean_text)

# 3. Etiketleme
le = LabelEncoder()
df["label"] = le.fit_transform(df["cyberbullying_type"])

# 4. Eğitim ve Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 5. Vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))  # N-gram aralığını (1, 3) olarak değiştirdik
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Model Eğitimi
model = MultinomialNB(alpha=0.5)  # Alpha parametresini 0.5 yaparak modelin dengesini ayarladık
model.fit(X_train_tfidf, y_train)

# 7. Kaydet
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")  # Etiket çevirici

# 8. Modeli Değerlendir
print(df["cyberbullying_type"].value_counts())

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_tfidf)

# Doğruluk oranını yazdır
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk Oranı (Accuracy):", round(accuracy * 100, 2), "%")

# Detaylı sınıflandırma raporu
print("\nSınıflandırma Raporu:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
