from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load trained model and tokenizer
model_path = "deepseek-ai/DeepSeek-R1"
token = 'LAPY27ZYQ3UXD9N7'
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict function
def predict_stock_movement(stock_text):
    inputs = tokenizer(stock_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Up" if probs[0][1] > probs[0][0] else "Down"

# Test example
if __name__ == "__main__":
    test_text = "Stock: AAPL, Open: 185.3, Close: 186.5, Change: 0.65"
    prediction = predict_stock_movement(test_text)
    print(f"Predicted Movement: {prediction}")