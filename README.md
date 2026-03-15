# Deep_learning_Sentiment_Analysis
A Deep Learning powered Sentiment Analysis Web Application built using Bidirectional LSTM (BiLSTM), TensorFlow, Keras, and Flask. The system performs real-time review classification and bulk CSV sentiment prediction with summary analytics (deep-learning sentiment-analysis).Bilstm,tensorflow,,flask,nlp,machine-learning,ai-project python web-application




# Deep Learning Based Sentiment Analysis Web Application

##  Project Overview

This project implements a Deep Learning-based Sentiment Analysis system using a Bidirectional LSTM (BiLSTM) network. The application classifies user reviews as **Positive** or **Negative** and also supports bulk CSV sentiment prediction through a Flask web interface.

The system is built using TensorFlow, Keras, NLTK, and Flask.

---

##  Model Architecture

The deep learning model consists of:

- Embedding Layer (Vocabulary Size: 20,000)
- Bidirectional LSTM (64 units)
- Dense Layer (ReLU Activation)
- Output Layer (Sigmoid Activation)

**Training Configuration:**

- Loss Function: Binary Crossentropy  
- Optimizer: Adam (Learning rate = 0.001)  
- Validation Split: 20%  
- Epochs: 5  
- Batch Size: 64  

---

##  Features

✔ Text Cleaning using Regular Expressions  
✔ Stopword Removal with Negation Handling  
✔ Lemmatization using WordNet  
✔ CSV Bulk Sentiment Prediction  
✔ Summary Statistics Generation  
✔ Interactive Flask Web Interface  

---

##  Project Structure

Deep-Learning-Sentiment-Analysis/
│
├── app.py              # Flask Web Application  
├── train.py            # Model Training Script  
├── predict.py          # Prediction Logic  
├── preprocess.py       # Text Preprocessing  
├── requirements.txt    # Project Dependencies  
│
├── model/              # Saved Model & Tokenizer  
├── dataset/            # Training Dataset  
├── templates/          # HTML Templates  
└── README.md  

---

## ⚙ Installation

### 1️ Clone the Repository

git clone https://github.com/phanikumar124/Deep-Learning-Sentiment-Analysis.git  
cd Deep-Learning-Sentiment-Analysis  

### 2️ Install Dependencies

pip install -r requirements.txt  

---

##  Training the Model

Run the training script:

python train.py  

After training, the following files will be generated inside the `model/` directory:

- sentiment_model.h5  
- tokenizer.pkl  

---

##  Run the Web Application

Start the Flask application:

python app.py  

Open your browser and go to:

http://127.0.0.1:5000  

---

##  Example Prediction

Input:  
"I am very happy with this medicine."

Output:  
Sentiment: Positive  
Confidence: 0.91  

---

##  Dataset Information

The model is trained on a drug review dataset containing:

- User Reviews  
- Corresponding Ratings  

Sentiment Labeling Rule:
- Rating ≥ 7 → Positive  
- Rating < 7 → Negative  

---

##  Future Improvements

- Add Attention Mechanism  
- Increase Epochs for Better Accuracy  
- Deploy on Render / AWS  
- Add REST API Support  
- Dockerize the Application  

---

##  Author
Phani kumar,Koushik,Pavan Kalyan
B.Tech – Artificial Intelligence & Machine Learning  



## ⭐ If you like this project, please give it a star!
