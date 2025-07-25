import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    'text': [
        "Congratulations! You've won a $1000 gift card. Call now!",
        "Hey, are we still on for lunch?",
        "You’ve been selected for a lottery prize. Claim now!",
        "Can we reschedule our meeting?",
        "Win a brand new iPhone now! Click the link.",
        "Don't forget about the assignment due tomorrow."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])  # Convert text to numbers
y = df['label']  # Labels: spam or not
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
your_email = ["You have a chance to win a free cruise. Click now!"]
your_email_vec = vectorizer.transform(your_email)
print("Spam" if model.predict(your_email_vec)[0] == 1 else "Not Spam")
