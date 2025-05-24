# TwittoBERT
This project demonstrates a sentiment analysis pipeline built with DistilBERT, a lightweight transformer model developed by Hugging Face. The model was fine-tuned on a dataset of 16,000 tweets to classify sentiment into categories such as Positive, Negative, and Neutral. The final model achieved an impressive 90% accuracy on the validation set.

🚀 Features
Utilizes DistilBERT for high-performance NLP with lower resource consumption.
Cleaned and preprocessed Twitter data (16K rows).
Fine-tuned with PyTorch and Hugging Face Transformers.
Achieved 90%+ accuracy on sentiment classification.
Includes training, validation, and evaluation pipelines.
📁 Dataset
16,000 manually labeled tweets with three sentiment classes:

Positive
Negative
Neutral
Dataset was preprocessed to remove mentions, hashtags, links, and special characters.

🧠 Model
Base Model: distilbert-base-uncased
Fine-tuning: Trained for several epochs using a cross-entropy loss function and AdamW optimizer.
Tokenizer: Hugging Face DistilBertTokenizerFast
Training Framework: PyTorch + Hugging Face Trainer API
📊 Performance
Metric	Score
Accuracy	90%
Precision	High
Recall	High
F1-score	High
Note: Actual precision, recall, and F1-score values can be added if available.

📦 Dependencies
transformers==4.x.x
torch==1.x
scikit-learn
pandas
numpy
matplotlib
Install with:

pip install -r requirements.txt
🛠️ How to Run
Clone the repository:

git clone https://github.com/yourusername/twitter-sentiment-distilbert.git
cd twitter-sentiment-distilbert
Install dependencies:

pip install -r requirements.txt
Train the model:

python train.py
Evaluate the model:

python evaluate.py
Run prediction on new tweets:

python predict.py --text "I love this app!"
📈 Example Output
Input: "I love this app!"
Predicted Sentiment: Positive
📚 Future Improvements
Integrate with a live Twitter API for real-time sentiment tracking.
Add a web dashboard using Streamlit or Flask.
Extend to multilingual support using xlm-roberta.
📄 License
This project is open-source and available under the MIT License.
