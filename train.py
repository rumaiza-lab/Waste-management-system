#Training Model for classification
#!/usr/bin/env python3
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time, os, copy
# Define the path to your dataset
data_dir = "archive (1)/Dataset"
# Data augmentation and normalization for training; only normalization for validation
data_transforms = {
'train': transforms.Compose([
transforms.Resize(256),
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
]),
for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print("Classes detected:", class_names)
# Expecting something like: ['biodegradable', 'non-biodegradable']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
for epoch in range(num_epochs):
print('-' * 40)
print(f'Epoch {epoch+1}/{num_epochs}')
# Each epoch has a training and validation phase.

for phase in ['train', 'val']:
if phase == 'train':
model.train() # Set model to training mode.
else:
model.eval() # Set model to evaluation mode.
running_loss = 0.0
running_corrects = 0
# Iterate over data.
for inputs, labels in dataloaders[phase]:
inputs = inputs.to(device)
labels = labels.to(device)
optimizer.zero_grad()
# Forward pass
with torch.set_grad_enabled(phase == 'train'):
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
loss = criterion(outputs, labels)
# Backward pass and optimize only if in training phase.
if phase == 'train':
loss.backward()
optimizer.step()
running_loss += loss.item() * inputs.size(0)
running_corrects += torch.sum(preds == labels.data)
if phase == 'train':
scheduler.step()
epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects.double() / dataset_sizes[phase]
print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
# Deep copy the model if validation accuracy improves.
if phase == 'val' and epoch_acc > best_acc:
best_acc = epoch_acc
best_model_wts = copy.deepcopy(model.state_dict())
time_elapsed = time.time() - since
print('-' * 40)
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:.4f}')
model.load_state_dict(best_model_wts)
return model
def main():
# Load a pretrained MobileNetV2 model
model_ft = models.mobilenet_v2(pretrained=True)
# Replace the classifier to match our two classes
model_ft.classifier[1] = nn.Linear(model_ft.last_channel, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=25)
# Save the trained model weights
torch.save(model_ft.state_dict(), "waste_classifier.pth")
print("Model saved as waste_classifier.pth")
if __name__ == '__main__':
main()
