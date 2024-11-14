import torch
from torch import nn, optim
from torch.utils.data import Subset
from torchvision import datasets, models, transforms
import random
import pickle
import os

# -- Define data transformations
slight_rotation = transforms.RandomRotation(degrees=(-10,10)) #rotation between -10 and 10 degrees
slight_translation = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Translate up to 10% horizontally and vertically

data_transforms = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 pixels
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomApply([slight_rotation], p = 0.5), #apply rotation 50% of the time
    transforms.RandomApply([slight_translation], p = 0.5), #apply translation 50% of the time
    # Normalize using ImageNet's mean and standard deviation
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# -- Load piece chip dataset
dataset = datasets.ImageFolder('archive/segmented_train/', transform=data_transforms)

# -- Randomly sample 40000 empty square samples to reduce dataset size
print("Reducing empty samples...")
empty_idx = 0
empty_samples = 120000
load_indices = True
class_indices_file = 'new_class_indices.pkl'

# -- Load new_class_indices if available, otherwise generate and save them
if load_indices and os.path.exists(class_indices_file):
    with open('new_class_indices.pkl', 'rb') as f:
        new_class_indices = pickle.load(f)
else:
    print("-- Finding all empty indices...")
    all_empty_indices = [i for i, label in enumerate(dataset.targets) if label == empty_idx]
    print("-- Randomly selecting new empty indices...")
    new_empty_indices = random.sample(all_empty_indices, empty_samples)
    print("-- Creating new class indices...")
    new_class_indices = [i for i, label in enumerate(dataset.targets) if label != empty_idx or i in new_empty_indices]

    # Save new class indices for reuse
    with open('new_class_indices.pkl', 'wb') as f:
        pickle.dump(new_class_indices, f)

# -- Create a subset of the dataset with few_empty_indices instead of all_empty_indices (new_class_indices)
reduced_dataset = Subset(dataset, new_class_indices)

# -- Create DataLoader
dataloader = torch.utils.data.DataLoader(reduced_dataset, batch_size=32, shuffle=True)

# -- Load pretrained ResNet-18 model
print("\nLoading ResNet-18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print("Model loaded successfully.")

# -- Adjust the final layer to number of classes (13)
model.fc = nn.Linear(model.fc.in_features, 13)

# -- Set model device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# -- Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -- Train the model
print("\nTraining...")

model.train()

num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"-- Batch {i} loss: {loss.item():.4f}")

    print(f"- Epoch {epoch} loss: {loss.item():.4f}")
    
    torch.save(model, f'classifier_epoch{epoch}.pth')
    print("- Saved model successfully.\n")