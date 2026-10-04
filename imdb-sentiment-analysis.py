import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(filepath):
    """ Φορτώνει το dataset και αφαιρεί duplicates """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    return df


def clean_text(text):
    """ Καθαρίζει το κείμενο: (αφαιρεί HTML tags, σημεία στίξης, μετατρέπει σε πεζά) """
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_data(df):
    """ Εφαρμόζει καθαρισμό κειμένου και μετατρέπει τα labels σε 0/1 """
    df['clean_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df


def split_data(df, test_size=0.2, random_state=42):
    """ Χωρίζει τα δεδομένα σε train/test sets """
    X = df['clean_review']
    y = df['label']
    return train_test_split(X, y, test_size=test_size,
                             random_state=random_state, stratify=y)


def vectorize_text(X_train, X_test, max_features=5000):
    """ Μετατρέπει κείμενο σε TF-IDF vectors """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_model(X_train_tfidf, y_train):
    """ Εκπαιδεύει το Logistic Regression μοντέλο """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model


def evaluate_model(model, X_test_tfidf, y_test):
    """ Αξιολογεί το μοντέλο και τυπώνει metrics """
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def main():
    
    print("Φόρτωση δεδομένων...")
    df = load_data('IMDB Dataset.csv')

    print("Καθαρισμός κειμένου...")
    df = preprocess_data(df)

    print("Διαχωρισμός σε train/test...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("TF-IDF vectorization...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)


    print("Εκπαίδευση μοντέλου...")
    model = train_model(X_train_tfidf, y_train)

    print("Αξιολόγηση μοντέλου...\n")
    evaluate_model(model, X_test_tfidf, y_test)
    

if __name__ == '__main__':
    main()