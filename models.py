import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
print("heyy i run")
# Custom Dataset for YOLO formatted data
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                values = line.strip().split()
            if len(values) < 5:
                raise ValueError(f"Invalid annotation line (less than 5 values): {values}")
            elif len(values) > 5:
                print(f"Warning: Extra values detected in {label_path}, ignoring extra values: {values}")
                values = values[:5]  # Only take the first 5 values
        
            class_id, x_center, y_center, bw, bh = map(float, values)

        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Empty tensor
            labels = torch.zeros((0,), dtype=torch.int64)  # No labels

        
        target = {'boxes': boxes, 'labels': labels}
        return image, target

# Define transformations
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Load dataset
dataset = YOLODataset("E:/Car_Parking_Detection_System/dataset/final/images/train", "E:/Car_Parking_Detection_System/dataset/final/labels/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print(f"Dataset size: {len(dataset)}")

# Model selection
def get_model(model_name):
    if model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "retinanet":
        model = retinanet_resnet50_fpn(pretrained=True)
    elif model_name == "maskrcnn":
        model = maskrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "detr":
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    else:
        raise ValueError("Model not supported yet!")
    return model

# Training loop
def train_model(model, dataloader, epochs=10, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Example usage
model = get_model("fasterrcnn")  # Change to retinanet, detr, maskrcnn
train_model(model, dataloader)
