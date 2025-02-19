from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from datasets import Dataset

# Load the trained model and tokenizer
model_name = "deepseek_stock_model"  # Path to the saved model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load some new data for testing (example stock data)
new_data = pd.DataFrame({
    "Open": [150.25, 200.75],  # Example stock opening prices
    "Close": [155.50, 205.60],  # Example stock closing prices
    "Price Change %": [3.50, 2.45],  # Example price change percentages
})

# Convert data into text format for prediction
def format_data_for_prediction(df):
    return [
        f"Stock: AAPL, Open: {row.Open}, Close: {row.Close}, Change: {row['Price Change %']:.2f}"
        for _, row in df.iterrows()
    ]

# Format the new data
formatted_data = format_data_for_prediction(new_data)

# Tokenize the new data
inputs = tokenizer(formatted_data, padding=True, truncation=True, return_tensors="pt", max_length=128)

# Move inputs to the same device as model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Get model predictions
with torch.no_grad():  # Disable gradient calculation for inference
    model.eval()  # Set model to evaluation mode
    outputs = model(**inputs)

# Get predictions (using argmax for classification)
predictions = torch.argmax(outputs.logits, dim=-1)

# Print the predictions (0 or 1)
print("Predictions:", predictions)

# Map predictions to actual labels (if necessary)
# Assuming 0 = "Down" and 1 = "Up" for stock prediction
labels = ["Down", "Up"]
predicted_labels = [labels[pred] for pred in predictions]

print("Predicted stock movements:", predicted_labels)
