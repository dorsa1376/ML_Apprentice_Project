import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# Load a pre-trained transformer model for generating embeddings
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a custom model class that supports multi-task learning
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model  # Save the pre-trained transformer as an attribute

        # Define the binary classification head for Task A (positive vs. negative)
        self.classifier_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Define the sentiment analysis head for Task B (positive, neutral, negative)
        self.sentiment_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        # Define separate loss functions for each task
        self.loss_fn_A = nn.CrossEntropyLoss()
        self.loss_fn_B = nn.CrossEntropyLoss()

    # Define the forward pass to compute predictions
    def forward(self, sentences, labels_A=None, labels_B=None):
        embeddings = torch.tensor(self.base_model.encode(sentences))
        classification_output = self.classifier_head(embeddings)
        sentiment_output = self.sentiment_head(embeddings)

        if labels_A is not None and labels_B is not None:
            loss_A = self.loss_fn_A(classification_output, labels_A)
            loss_B = self.loss_fn_B(sentiment_output, labels_B)
            total_loss = (loss_A + loss_B) / 2
            return classification_output, sentiment_output, total_loss

        return classification_output, sentiment_output

# Create an instance of the model
multi_task_model = MultiTaskModel(base_model)

# Freeze transformer layers
for param in base_model.parameters():
    param.requires_grad = False

# Define an optimizer for trainable parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, multi_task_model.parameters()), lr=0.001)

# Fake training data (for demonstration)
train_sentences = [
    "I love exploring space.",
    "This is a terrible mistake.",
    "The weather is okay today.",
    "I absolutely hate waiting in line."
]
labels_A = torch.tensor([1, 0, 1, 0])  # Binary labels (positive or negative)
labels_B = torch.tensor([2, 0, 1, 0])  # Multi-class labels (positive, neutral, negative)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    _, _, total_loss = multi_task_model(train_sentences, labels_A, labels_B)
    total_loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")

# Save the trained model
torch.save(multi_task_model.state_dict(), 'multi_task_model_heads.pth')
print("Model saved as multi_task_model_heads.pth")
