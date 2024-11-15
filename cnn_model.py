import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

# Load datasets
from torchvision import datasets
import torchvision.transforms as transforms

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Prepare dataloaders
dataloader_train = DataLoader(train_data, shuffle=True, batch_size=64)
dataloader_test = DataLoader(test_data, shuffle=False, batch_size=64)


# Define model
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# Define loss function and optimizer
net = Net(num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train model
for epoch in range(1):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate model
metric_precision = Precision(task="multiclass", num_classes=10, average="macro")
metric_recall = Recall(task="multiclass", num_classes=10, average="macro")
metric_accuracy = Accuracy(task="multiclass", num_classes=10)

metric_precision.reset()
metric_recall.reset()
metric_accuracy.reset()

net.eval()

predictions = []

metric_precision = Precision(task="multiclass", num_classes=10, average=None)
metric_recall = Recall(task="multiclass", num_classes=10, average=None)
metric_accuracy = Accuracy(task="multiclass", num_classes=10)

with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)

        metric_precision.update(preds, labels)
        metric_recall.update(preds, labels)
        metric_accuracy.update(preds, labels)

        predictions.extend(preds.tolist())

precision = metric_precision.compute().tolist()
recall = metric_recall.compute().tolist()
accuracy = float(metric_accuracy.compute())

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
