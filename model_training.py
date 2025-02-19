from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import Dataset

# Load DeepSeek model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model without applying classification head directly
try:
    model = AutoModel.from_pretrained(model_name)
    print("DeepSeek Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check if model is loaded
if model is None:
    raise ValueError("Model failed to load. Please check the model name and your connection.")


# Load preprocessed data
df = pd.read_csv("stock_data.csv")

# Convert data into text format for model training
def format_data(df):
    return {
        "text": [
            f"Stock: AAPL, Open: {row.Open}, Close: {row.Close}, Change: {row['Price Change %']:.2f}"
            for _, row in df.iterrows()
        ],
        "label": df["Target"].tolist(),
    }

dataset = Dataset.from_dict(format_data(df))

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./deepseek_stock_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    processing_class=tokenizer,
)

trainer.train()

# Save trained model
model.save_pretrained("deepseek_stock_model")
tokenizer.save_pretrained("deepseek_stock_model")
print("Model training complete!")
