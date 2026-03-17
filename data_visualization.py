# Data Visualization Script for Tiny ImageNet Dataset
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for the dataset (same as in train.py)
transform = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load the dataset (assuming it's already downloaded in 'dataset' folder)
dataset_path = 'dataset/tiny-imagenet-200'
if os.path.exists(dataset_path):
    tiny_imagenet_dataset_train = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform['train'])
    tiny_imagenet_dataset_test = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform['test'])

    # Create a DataLoader
    dataloader_train = DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(tiny_imagenet_dataset_test, batch_size=32, shuffle=False)

    # Function to denormalize image for visualization
    def denormalize(image):
        image = image.to('cpu').numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        return image

    # Visualize one example for each class for 10 classes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes_sampled = []
    found_classes = 0

    for inputs, labels in dataloader_train:
        for j in range(len(inputs)):
            class_id = labels[j].item()

            if class_id not in classes_sampled and found_classes < 10:
                img = denormalize(inputs[j])

                ax = axes[found_classes // 5, found_classes % 5]
                ax.imshow(img)
                ax.set_title(f"Class: {class_id}")
                ax.axis('off')

                classes_sampled.append(class_id)
                found_classes += 1

            if found_classes >= 10:
                break
        if found_classes >= 10:
            break

    plt.tight_layout()
    plt.savefig('data_visualization.png')  # Save the plot instead of showing
    print("Visualization saved as 'data_visualization.png'")
else:
    print("Dataset not found. Please run train.py first to download the dataset.")