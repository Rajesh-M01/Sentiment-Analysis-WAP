# Sentiment-Analysis-WAP
Sentiment Analysis Using Naïve Bayes and BERT
This project implements a hybrid sentiment analysis model using Naïve Bayes (for traditional machine learning) and BERT (for deep learning). The model classifies text into three sentiment categories: Positive, Neutral, and Negative. The hybrid approach combines the simplicity of Naïve Bayes for feature extraction and the power of BERT for contextual understanding.

Dataset:
The dataset used for training the sentiment analysis model is sourced from Kaggle and contains customer feedback reviews. Each review is labeled as positive, neutral, or negative. 

Model Overview:
Preprocessing:
•	Text Cleaning: Removing stopwords, special characters, and normalizing the text.
•	Tokenization: Converting text into tokens for processing.

Architecture:
•	Naïve Bayes:
o	Utilized for traditional sentiment classification through probabilistic models with TF-IDF vectorization.
•	BERT:
o	A pre-trained model from Hugging Face’s Transformers library, fine-tuned for sentiment classification to understand the context of the text

Training:
•	Loss Function: Cross-entropy loss function for classification.
•	Optimizer: Adam optimizer for better convergence.

Results:
BERT ->
Final BERT Test Accuracy: 89.69%
![image](https://github.com/user-attachments/assets/1d912f3e-3eb6-4f1c-afa6-59648d547e7a)

NAIVE BAYES ->
Naive Bayes Accuracy:  75.29137529137529%
![image](https://github.com/user-attachments/assets/f5b0097b-5f7b-4339-8624-b967036903bb)

Confusion Matrix ->
![image](https://github.com/user-attachments/assets/bad75888-3203-4322-aadb-a7bb4141d098)


SAMPLE INPUT AND OUTPUT->
1. Predicted Sentiment for I love this product! It's amazing.  ->  Positive 
2. Predicted Sentiment for Disgusting Product ! Would never recommend it.  ->  Negative 
3. Predicted Sentiment for It's decent ! Not Bad Not Good  ->  Neutral 

How to Run:
git clone https://github.com/Rajesh-M01/Sentiment-Analysis-WAP.git
cd Sentiment-Analysis-WAP

Run the Jupyter Notebook (Sentiment-Analysis-WAP.ipynb) in Kaggle through the link -> https://www.kaggle.com/code/rajeshm0/sentiment-analysis1

Libraries Used :
Tensorflow/Keras
Numpy
Pandas
Matplotlib
Seaborn
Transformers
Scikit-learn

Key Learnings
How to combine Naïve Bayes and BERT for sentiment analysis to leverage both traditional ML and deep learning models.
The importance of text preprocessing (tokenization, stopword removal) for enhancing model performance in NLP tasks.
How to handle imbalanced datasets by upsampling and use metrics like accuracy, precision, recall, and F1-score for model evaluation.
The effectiveness of hybrid models (Naïve Bayes + BERT) in improving sentiment classification accuracy compared to individual models.


🚀 Connect with Me
If you liked this project, feel free to ⭐ the repository and connect with me on https://www.linkedin.com/in/rajesh-m-a42539317/ ! 🚀





