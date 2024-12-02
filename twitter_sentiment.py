import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load Data
train = pd.read_csv(r'C:\Users\hp\OneDrive - Graphic Era University\Desktop\projects\Twitter-Sentiment-Analysis-master\train_tweet.csv')
test = pd.read_csv(r'C:\Users\hp\OneDrive - Graphic Era University\Desktop\projects\Twitter-Sentiment-Analysis-master\test_tweets.csv')

# Print data shapes
print(train.shape)
print(test.shape)

# Check for missing values
print(train.isnull().any())
print(test.isnull().any())

# Visualizing class distribution
train['label'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))

# Check length distribution
train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()
train['len'].plot.hist(color='pink', figsize=(6, 4), title='Tweet Length Distribution')

# Preprocessing - Text Cleaning & Tokenization
nltk.download('stopwords')

def clean_text(text):
    """Function to clean and preprocess text."""
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Remove stopwords and perform stemming
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

train['clean_tweet'] = train['tweet'].apply(clean_text)
test['clean_tweet'] = test['tweet'].apply(clean_text)

# Word Frequency Visualization
cv = CountVectorizer(max_features=2500)
x_train = cv.fit_transform(train['clean_tweet']).toarray()
x_test = cv.transform(test['clean_tweet']).toarray()

# Target variable
y_train = train['label']


# Split train data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# Standardization
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

# Model Evaluation
def evaluate_model(model, x_train, y_train, x_valid, y_valid):
    """Train, predict, and evaluate the given model."""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    print(f"Training Accuracy: {model.score(x_train, y_train)}")
    print(f"Validation Accuracy: {model.score(x_valid, y_valid)}")
    print(f"F1 Score: {f1_score(y_valid, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_valid, y_pred)}")
    print("-" * 50)

# Model Testing - RandomForest, LogisticRegression, SVM, XGBoost
models = [
    ("Random Forest", RandomForestClassifier()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Decision Tree", DecisionTreeClassifier()),
    ("SVM", SVC()),
    ("XGBoost", XGBClassifier())
]

# Evaluate each model
for model_name, model in models:
    print(f"Evaluating {model_name}")
    evaluate_model(model, x_train, y_train, x_valid, y_valid)

# Train the final model (You can choose one from above models, e.g., Random Forest)
final_model = RandomForestClassifier()
final_model.fit(x_train, y_train)

# Predict on test data
predictions = final_model.predict(x_test)

# Create the output DataFrame for predictions
output = pd.DataFrame({
    'Tweet': test['tweet'],  # Tweet column from test dataset
    'Predicted Sentiment': ['Positive' if sentiment == 1 else 'Negative' for sentiment in predictions]  # Mapping numerical predictions to sentiment labels
})

# Save the predictions to CSV
output.to_csv('predictions_output.csv', index=False)

# Print the first few rows of the output
print(output.head())
