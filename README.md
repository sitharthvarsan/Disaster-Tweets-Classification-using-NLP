 Disaster Tweets Classification using NLP, BERT & SBERT

This project classifies tweets as **real disaster-related** or **not** using advanced **NLP techniques**, including traditional vectorization methods (TF-IDF, Word2Vec, GloVe) and **transformer-based models like BERT.



 Project Overview

- Dataset: Tweets labeled as disaster (1) or non-disaster (0)
- Goal: Build models that accurately classify whether a tweet refers to a real disaster
- Techniques Used:
  - Text preprocessing & cleaning
  - Tokenization & Embeddings (TF-IDF, Word2Vec, GloVe, BERT)
  - Traditional ML models (Logistic Regression, LightGBM)
  - Deep Learning (BiLSTM)
  - Transformer-based models (BERT fine-tuning)



 Dataset

Available from [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview)

- `train.csv`: Contains tweets and labels
- `test.csv`: Test set for predictions
- `sample_submission.csv`: Submission format for Kaggle



 Preprocessing

- Remove URLs, HTML tags, punctuations, and stopwords
- Lowercase text
- Optional: Lemmatization

---

 Feature Extraction

1. TF-IDF 
   - Converts tweets into numeric vectors based on term importance  
   - Good for traditional ML models

2. Word2Vec 
   - Embeds each word in a vector space using context
   - Sentence vectors formed by averaging word vectors

3. GloVe 
   - Captures global word-word co-occurrence statistics
   - Pretrained vectors used for downstream models

4. BERT Embeddings  
   - Context-aware word/sentence embeddings
   - Uses [CLS] token for classification



  Models

- **Logistic Regression + TF-IDF**
- **LightGBM + TF-IDF/Word2Vec/GloVe**
- **BiLSTM + Word2Vec/GloVe**
- **BERT (fine-tuned)**



 BERT Fine-Tuning

- Pretrained BERT from `transformers` (e.g., `bert-base-uncased`)
- Fine-tuned on disaster tweet dataset
- Uses [CLS] token’s output for classification
- Classification head: Dense → ReLU → Dropout → Softmax



 How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run preprocessing:
   ```bash
   python preprocess.py
   ```

3. Train BERT model:
   ```bash
   python train_bert.py
   ```

4. Evaluate:
   ```bash
   python evaluate.py
   ```



 Evaluation Metrics

- Accuracy
- Precision / Recall / F1 Score
- Confusion Matrix
- ROC-AUC (optional)

---

 Highlights

- Handles Out-of-Vocabulary (OOV) words using BERT subword tokenization
- Explored both classical ML and deep learning approaches
- BERT significantly improves classification performance due to contextual understanding

---

 Future Improvements

- Hyperparameter tuning
- Ensemble multiple models
- Deploy with FastAPI or Flask
- Interpretability using SHAP or LIME



 Acknowledgements

- [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- HuggingFace Transformers
- spaCy, NLTK, Scikit-learn

