import zipfile
import gdown
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Model klasörü (model.zip aynı dizine indirilecek)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

if not os.path.exists(MODEL_DIR):
    print("Model not found. Downloading from Google Drive...")
    zip_path = os.path.join(os.path.dirname(__file__), "model.zip")
    gdown.download("https://drive.google.com/uc?id=1isfRttf8iNomlKil8SYEsCb9VuOWdqin", zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    print("Model downloaded and extracted.")

model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model.eval()

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    lang = data.get("lang", "tr")  # varsayılan Türkçe
    if lang == "tr":
        text = GoogleTranslator(source='tr', target='en').translate(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)

    # Tüm olasılıkları liste olarak döndür
    emotion_scores = {emotion_labels[i]: round(float(probs[0][i]) * 100, 2) for i in range(len(emotion_labels))}

    # 50% üzerindekileri ayrıca çıkar (opsiyonel)
    predicted = [label for label, score in emotion_scores.items() if score > 50]


    return jsonify({
        "predicted_emotions": predicted,
        "emotion_scores": emotion_scores
    })

# if __name__ == "__main__":
#     app.run(debug=True)