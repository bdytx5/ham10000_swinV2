import torch
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import wandb
from transformers import AutoModel, Swinv2Model, AutoImageProcessor, MobileViTFeatureExtractor, MobileViTModel
import torch.nn as nn
import random
from torchvision import transforms
import datetime
from torch.optim import Adam


class HAM10000DatasetBalanced(Dataset):
    def __init__(self, csv_file, img_dir, augment=True, max_per_class=100000):

        self.skin_df = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.augment = augment

        self.transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomRotation(20),  # Random rotation between -20 to 20 degrees
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
        self.base_transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])        
    def __len__(self):
        return len(self.skin_df)

    def label_to_int(self, label):
        label_dict = {
            'nv': 0,      # Melanocytic nevi
            'mel': 1,     # Melanom`a
            'bkl': 2,     # Benign keratosis-like lesions
            'bcc': 3,     # Basal cell carcinoma
            'akiec': 4,   # Actinic keratoses
            'vasc': 5,    # Vascular lesions
            'df': 6       # Dermatofibroma
        }
        return label_dict.get(label, -1)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            try:
                img_name = os.path.join(self.img_dir, self.skin_df.iloc[idx, 1] + '.jpg')
                image = Image.open(img_name)
                if self.augment:
                    image = self.transform(image)
                else: 
                    image = self.base_transform(image)
                # image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
                label = self.label_to_int(self.skin_df.iloc[idx, 2])
                return image, label

            except (FileNotFoundError, UnidentifiedImageError):
                print(f"Error opening image: {img_name}. Trying another image.")
                attempts += 1
                idx = random.randint(0, len(self.skin_df) - 1)

        raise Exception(f"Failed to load an image after {max_attempts} attempts.")

class SwinV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Classifier, self).__init__()
        self.model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        # self.classifier = nn.Linear(64 * 768, num_classes)218880
        self.classifier = nn.Linear(218880, num_classes)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state.flatten(start_dim=1)
        return self.classifier(features)





tr_ds = HAM10000DatasetBalanced(csv_file="/home/brett/Desktop/ham10000/HAM10000_metadata.csv", img_dir="/home/brett/Desktop/ham10000/imgs")
tst_ds = HAM10000DatasetBalanced(csv_file="/home/brett/Desktop/ham10000/ISIC2018_Task3_Test_GroundTruth.csv", img_dir="/home/brett/Desktop/ham10000/imgs", augment=False)



def make_weights_for_balanced_classes(dataset):
    class_counts = dataset.skin_df['dx'].value_counts()
    num_samples = len(dataset)
    class_weights = {i: num_samples/class_counts[i] for i in range(len(class_counts))}
    weights = [class_weights[label] for label in dataset.skin_df['dx'].map(dataset.label_to_int)]
    return weights

# Calculate weights for balanced sampling
weights = make_weights_for_balanced_classes(tr_ds)
sampler = WeightedRandomSampler(weights, len(weights))

# Create your DataLoaders
train_loader = DataLoader(tr_ds, batch_size=12, sampler=sampler)  # Use sampler here

tst_loader = DataLoader(tst_ds, batch_size=12 )



def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy




def evaluate(model, data_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    return val_loss / len(data_loader), accuracy, all_preds, all_labels




model = SwinV2Classifier(num_classes=7)
model = model.cuda()



current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = f'./runs/run_{current_time}'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


criterion = nn.CrossEntropyLoss()
epochs = 1000
lr = 0.000005
optimizer = Adam(model.parameters(), lr=lr)

best_train_loss = float('inf')
best_val_accuracy = 0.0
wandb.init(project="skin-lesion-classification")


label_dict = {
            'nv': 0,      # Melanocytic nevi
            'mel': 1,     # Melanom`a
            'bkl': 2,     # Benign keratosis-like lesions
            'bcc': 3,     # Basal cell carcinoma
            'akiec': 4,   # Actinic keratoses
            'vasc': 5,    # Vascular lesions
            'df': 6       # Dermatofibroma
        }


class_names = [key for key, value in sorted(label_dict.items(), key=lambda item: item[1])]

for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy, val_preds, val_labels = evaluate(model, tst_loader, criterion)

    print(f"LR: {lr}, Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    wandb.log({"lr": lr, "epoch": epoch+1, "train_loss": train_loss,"train_accuracy": train_accuracy, "test_loss": val_loss, "test_accuracy": val_accuracy})
    wandb.log({"conf_mat_test": wandb.plot.confusion_matrix(probs=None, y_true=val_labels, preds=val_preds, class_names=class_names)})

    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), os.path.join(models_dir, f'best_train_model_lr{lr}.pth'))

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), os.path.join(models_dir, f'best_val_model_lr{lr}.pth'))

wandb.finish()

