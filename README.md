# Twitter Sentiment Analysis

This project aims to analyze the sentiment of tweets and classify them into two categories: **Racist/Sexist** and **Non-Racist/Sexist**. The analysis involves extracting hashtags, tokenizing the tweets, and building a Word2Vec model to understand the contextual relationships between words in the tweets.

## Project Overview

The goal of this project is to classify tweets as either racist/sexist or non-racist/sexist. The sentiment classification is achieved by using various natural language processing (NLP) techniques such as tokenization, word embeddings (Word2Vec), and hashtag extraction.

## Steps Involved

1. **Data Preprocessing**: 
   - Load and clean the dataset.
   - Handle missing values, duplicate entries, and irrelevant columns.
   - Perform text cleaning such as removing special characters, converting text to lowercase, etc.

2. **Hashtag Extraction**:
   - Extract hashtags from tweets labeled as non-racist/sexist and racist/sexist using regular expressions.

3. **Text Tokenization**:
   - Tokenize tweets into individual words for further analysis and model training.

4. **Word2Vec Model**:
   - Use the Word2Vec model to convert words in tweets into vector representations (word embeddings).
   - Train the Word2Vec model on the tokenized tweets.

5. **Model Evaluation**:
   - Evaluate the model's performance using classification metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used

- **Python**: Programming language used for data analysis and model building.
- **Pandas**: Library for data manipulation and analysis.
- **NumPy**: Library for numerical computing.
- **Gensim**: Library for creating and training the Word2Vec model.
- **Matplotlib / Seaborn**: Libraries for visualizing the data and results.
- **Scikit-learn**: Library for evaluating the model's performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   
2. Navigate to the project directory:
   cd twitter-sentiment-analysis
   
4. Create a virtual environment and activate it:
    sentiment_analysis_env\Scripts\activate

## File Structure
Twitter-Sentiment-Analysis/
├── data/
│   └── train.csv  # Training data
├── models/
│   └── model_w2v  # Trained Word2Vec model
├── notebooks/
│   └── analysis.ipynb  # Jupyter notebook for data analysis and model training
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preprocessing functions
│   ├── feature_extraction.py  # Functions for hashtag extraction and tokenization
│   └── model.py  # Code for training and evaluating the model
├── requirements.txt  # List of required Python packages
└── README.md  # Project documentation
## Evaluation Metrics:

After training the model, the following classification metrics are evaluated:

Accuracy
Precision
Recall
F1-Score

## Contributing:
Feel free to fork this repository, make improvements, and submit pull requests. If you find any issues or bugs, please create an issue in the repository.
