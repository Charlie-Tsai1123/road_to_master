# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, model_name, num_epochs=10):
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        val_accuracies.append(acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {acc:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'model/{model_name}.pth')
    
    return train_losses, val_accuracies, best_acc

def main():
    # Data transforms without random erasing
    base_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
        # Data transforms with random erasing
    random_erase_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # Convert image to tensor first
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()  # Apply RandomErasing after ToTensor
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Train first model (without random erasing)
    print("Training model without random erasing...")
    train_dataset = datasets.ImageFolder('dataset/train', base_transforms)
    val_dataset = datasets.ImageFolder('dataset/val', val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model1 = models.resnet50(pretrained=True)
    num_ftrs = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs, 2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    
    _, _, acc1 = train_model(model1, train_loader, val_loader, criterion, optimizer, 
                            "resnet50_without_random_erasing")
    
    # Train second model (with random erasing)
    print("\nTraining model with random erasing...")
    train_dataset = datasets.ImageFolder('dataset/train', random_erase_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model2 = models.resnet50(pretrained=True)
    num_ftrs = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs, 2)
    
    optimizer = optim.Adam(model2.parameters(), lr=0.001)
    
    _, _, acc2 = train_model(model2, train_loader, val_loader, criterion, optimizer,
                            "resnet50_with_random_erasing")
    
    # Save accuracies for comparison
    accuracies = {
        'Without Random Erasing': acc1,
        'With Random Erasing': acc2
    }
    
    # Plot accuracy comparison
    # Plot accuracy comparison
    plt.figure(figsize=(8, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values())

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')

    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig('accuracy_comparison.png')
    plt.show()
    
    # Save which model is better
    with open('model/best_model.txt', 'w') as f:
        if acc2 > acc1:
            f.write('resnet50_with_random_erasing.pth')
        else:
            f.write('resnet50_without_random_erasing.pth')

if __name__ == '__main__':
    main()
