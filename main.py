import pathlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from PIL import Image
from dataset import ImageDataset
from torch.utils.data import DataLoader
from model import Model

base_dir = pathlib.Path("main_program/datasets")
folders = ["robert_downey_jr", "megan_fox"]
train_file_list = []
test_file_list = []
train_labels = []
test_labels = []
for folder in folders:
    imgdir_path = base_dir / folder
    individual_file_list = sorted([str(path) for path in imgdir_path.glob("*.png")])
    train_file_list.extend(individual_file_list[:80])
    test_file_list.extend(individual_file_list[80:])
    train_labels.extend(
        [
            1 if "robert" in os.path.dirname(file) else 0
            for file in individual_file_list[:80]
        ]
    )
    test_labels.extend(
        [
            1 if "robert" in os.path.dirname(file) else 0
            for file in individual_file_list[80:]
        ]
    )
# print(train_file_list)
# print(test_file_list)
# print(train_labels)
# print(test_labels)
img_height, img_width = 300, 300
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((img_height, img_width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_image_data = ImageDataset(train_file_list, train_labels, transform)
test_image_data = ImageDataset(test_file_list, test_labels, transform)

#  visualization of the retrieved images with their labels
#  fig = plt.figure(figsize=(20, 10))
#  for i, example in enumerate(image_dataset):
#      ax = fig.add_subplot(10, 20, i + 1)
#      ax.set_xticks([])
#      ax.set_yticks([])
#      ax.imshow(example[0].numpy().transpose((1, 2, 0)))
#     ax.set_title(f"{example[1]}", size=4)
#  plt.tight_layout()
#  plt.show()

train_dataloader = DataLoader(train_image_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_image_data, batch_size=5, shuffle=True)
input_size = 300 * 300 * 3
hidden_size = 128
output_size = 2
model = Model(input_size, hidden_size, output_size)
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 20
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    total_samples = 0
    for x_batch, y_batch in train_dataloader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        epoch_accuracy += is_correct.sum().item()
        total_samples += y_batch.size(0)
    epoch_loss /= total_samples
    epoch_accuracy /= total_samples
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
    )
    loss_hist[epoch] = epoch_loss
    accuracy_hist[epoch] = epoch_accuracy
# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot training accuracy
plt.figure(figsize=(10, 5))
plt.plot(accuracy_hist, label="Training Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate model
model.eval()  # Set the model to evaluation mode
test_loss = 0
test_accuracy = 0
total_test_samples = 0

with torch.no_grad():  # Disable gradient calculation
    for x_test, y_test in test_dataloader:
        test_pred = model(x_test)
        test_loss_batch = loss_fn(test_pred, y_test)
        test_loss += test_loss_batch.item() * y_test.size(0)

        # Calculate accuracy
        is_correct = (torch.argmax(test_pred, dim=1) == y_test).float()
        test_accuracy += is_correct.sum().item()
        total_test_samples += y_test.size(0)

# Average test loss and accuracy
test_loss /= total_test_samples
test_accuracy /= total_test_samples

# Print the final test results
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "main_program/model.pth")
