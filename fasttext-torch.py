# FastText-Style Text Classification for Large Datasets

# ## 1. Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score



# ## 2. Dataset Preparation
class FastTextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, ngram_range=(3, 4), min_freq=1):
        """
        Initialize the dataset with texts, labels, and optional vocabulary.
        """
        self.texts = texts
        self.labels = labels
        self.ngram_range = ngram_range
        self.min_freq = min_freq

        if vocab is None:
            self.vocab, self.label_map = self.build_vocab_and_labels(texts, labels)
        else:
            self.vocab, self.label_map = vocab

        self.encoded_texts = [self.text_to_ngrams(text) for text in texts]
        self.encoded_labels = [self.label_map[label] for label in labels]

    def build_vocab_and_labels(self, texts, labels):
        """
        Create vocab of n-grams and map labels to indices.
        """
        ngrams = list(chain.from_iterable(self.text_to_ngrams(text) for text in texts))
        ngram_counts = Counter(ngrams)
        vocab = {
            ngram: idx + 1
            for idx, (ngram, count) in enumerate(ngram_counts.items())
            if count >= self.min_freq
        }
        vocab["<pad>"] = 0  # Add padding token
        label_map = {label: idx for idx, label in enumerate(set(labels))}
        return vocab, label_map

    def text_to_ngrams(self, text):
        """
        Tokenize text into n-grams.
        """
        tokens = text.split()
        ngrams = []
        for token in tokens:
            token = f"<{token}>"
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams.extend([token[i : i + n] for i in range(len(token) - n + 1)])
        return ngrams

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single data sample.
        """
        text = self.encoded_texts[idx]
        label = self.encoded_labels[idx]
        return text, label

    def collate_fn(self, batch):
        """
        Collate function for padding and batching.
        """
        texts, labels = zip(*batch)
        max_length = max(len(text) for text in texts)
        padded_texts = [
            text + ["<pad>"] * (max_length - len(text)) for text in texts
        ]
        text_indices = torch.tensor(
            [[self.vocab.get(ngram, 0) for ngram in text] for text in padded_texts]
        )
        labels = torch.tensor(labels)
        return text_indices, labels

# ## 3. FastText Model
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(FastTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        doc_vector = embedded.mean(dim=1)  # Average over sequence length
        output = self.fc(doc_vector)  # Shape: (batch_size, num_classes)
        return output

# ## 4. Training and Evaluation
# ### Sample Dataset for Testing


df_train = pd.read_csv('dbpedia_train.csv',header=None)
df_train = df_train.sample(10000)


texts = df_train[2].tolist()
labels = df_train[0].tolist()

# print(f"texts : {len(texts)}")

# print(f"labels : {len(labels)}")

# texts = [
#     "I loved the movie",
#     "The plot was dull",
#     "Amazing direction and acting",
#     "Waste of my time",
#     "Outstanding performances",
#     "Awful acting",
#     "I threw up",
#     "Meh...",
#     "Amazingly bad acting",
#     "Surprisingly good",
#     "Conforting"
# ]
# labels = ["positive", "negative", "positive", "negative", "positive","negative","negative","negative","negative","positive","positive"]

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels
)

# Prepare datasets and DataLoaders
train_dataset = FastTextDataset(train_texts, train_labels)
val_dataset = FastTextDataset(val_texts, val_labels, vocab=(train_dataset.vocab, train_dataset.label_map))

train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4
)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=4
)

# Model parameters
vocab_size = len(train_dataset.vocab)
embed_dim = 50
num_classes = len(train_dataset.label_map)

# Initialize model, loss function, and optimizer
model = FastTextClassifier(vocab_size, embed_dim, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ### Training Loop
num_epochs = 20
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_dataloader:
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_dataloader)
    val_accuracy = correct / total

    train_losses.append(epoch_loss / len(train_dataloader))
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# ### Plot Training and Validation Metrics
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.title("Training and Validation Metrics")
plt.show()
plt.savefig('foo.png')

# ## 5. Prediction

def predict(text, model, dataset):
    model.eval()
    text_ngrams = dataset.text_to_ngrams(text)
    text_indices = torch.tensor(
        [[dataset.vocab.get(ngram, 0) for ngram in text_ngrams]]
    ).to(device)
    with torch.no_grad():
        outputs = model(text_indices)
        predicted_label = outputs.argmax(dim=1).item()
        return list(dataset.label_map.keys())[predicted_label]



df_test = pd.read_csv('dbpedia_test.csv',header=None)
print("**************************************************************")

# ## 5. Testing and Predictions
def predict(text, model, dataset):
    model.eval()
    text_ngrams = dataset.text_to_ngrams(text)
    text_indices = torch.tensor([[dataset.vocab.get(ngram, 0) for ngram in text_ngrams]])
    with torch.no_grad():
        outputs = model(text_indices)
        predicted_label = outputs.argmax(dim=1).item()
        return "".join(list(dataset.label_map.keys())[predicted_label])


def predict_series(serie, model, dataset):
    output = list("")
    for x in serie:
        output.append(predict(x, model, dataset))
    return output


# print(predict_series(df_prediction["description"],model,dataset))
# df_prediction = df_test.sample(20).assign(predicted=lambda x: predict(x[2],model,dataset))
df_prediction = df_test.rename(columns={0: "class", 1: "name", 2: "description"})
# print(df_prediction["description"].count())
df_prediction = df_prediction.assign(predicted=predict_series(df_prediction["description"], model, train_dataset))

y_true = df_prediction['class']
y_pred = df_prediction['predicted']


# Macro and Micro averaged Precision and Recall
macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
micro_precision = precision_score(y_true, y_pred, average='micro')
micro_recall = recall_score(y_true, y_pred, average='micro')
print(f"Macro Precision: {macro_precision:.2%}")
print(f"Macro Recall: {macro_recall:.2%}")
print(f"Micro Precision: {micro_precision:.2%}")
print(f"Micro Recall:{micro_recall:.2%}")
