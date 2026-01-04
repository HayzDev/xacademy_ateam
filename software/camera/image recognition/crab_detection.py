"""Detects crabs in images"""

import shutil
import time
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
from torchvision import models, transforms

device = torch.device("cpu")
# Returns a simple transform that converts a PIL image to a PyTorch tensor
def get_transform():
    return ToTensor()

class CocoDetectionDataset(Dataset):
    # Init function: loads annotation file and prepares list of image IDs
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)

    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)

        # Package everything into a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)

        return image, target

# Load training dataset with transform applied
train_dataset = CocoDetectionDataset(
    # Change this to your path
    image_dir="dataset_images/train", 
    annotation_path="dataset_images/train/_annotations.coco.json",
    transforms=get_transform()
)

# Load validation dataset with same transform
val_dataset = CocoDetectionDataset(
    image_dir="dataset_images/train/",
    annotation_path="dataset_images/train/_annotations.coco.json",
    transforms=get_transform()
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Get one batch from the DataLoader
images, targets = next(iter(train_loader))

# Convert PIL Image and draw annotations
"""
for i in range(len(images)):
    image = images[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    image = (image * 255).astype(np.uint8)  # Rescale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = targets[i]['boxes']
    labels = targets[i]['labels']

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Class {label.item()}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show image with boxes using matplotlib
    plt.figure(figsize=(16, 12))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Sample {i + 1}")
    plt.show()
"""

# Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Get the number of classes in the dataset (including background)
num_classes = len(train_dataset.coco.getCatIds()) + 1  # +1 for background class

# Get the number of input features for the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the classifier head with a new one for the custom dataset's classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the specified device (GPU or CPU)
model.to(device)

# Get parameters that require gradients (the model's trainable parameters)
params = [p for p in model.parameters() if p.requires_grad]

# Define the optimizer (Stochastic Gradient Descent) with learning rate, momentum, and weight decay
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def clear():
    os.system("cls")

def add_dataset_image(filename):
    annotations = json.load(open(r"dataset_images/train/_annotations.coco.json"))
    
    for image in annotations["images"]:
        if image["file_name"] == filename:
            print("Image already exists in dataset")
            return
    

    im = cv2.imread(filename)
    window_name = "Select area"
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    area = cv2.selectROI(window_name, im, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow(window_name)
    category = int(input("Enter category: "))

    annotations["images"].append({
        "id": len(annotations["images"]) + 1,
        "width": im.shape[1],
        "height": im.shape[0],
        "file_name": filename,
        "date_captured": "2025-12-14T15:32:00+00:00"
    })
    
    annotations["annotations"].append({
        "id": len(annotations["annotations"]) + 1,
        "image_id": len(annotations["images"]),
        "category_id": category,
        "segmentation": [],
        "area": area[2] * area[3],
        "bbox": [area[0], area[1], area[2], area[3]],
        "iscrowd": 0
    })

    json.dump(annotations, open(r"dataset_images/train/_annotations.coco.json", "w"))

    print("Success!")

def clear_dataset():
    annotations = json.load(open(r"dataset_images/train/_annotations.coco.json"))
    annotations["images"] = []
    annotations["annotations"] = []
    json.dump(annotations, open(r"dataset_images/train/_annotations.coco.json", "w"))

def select_folder():
    tk.Tk().withdraw()
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = askdirectory(parent=root, initialdir="./")
    root.destroy()
    return folder_path

def add_folder_images():
    folder_path = select_folder()
    if not folder_path:
        return

    dest_dir = os.path.abspath("dataset_images/train")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            src_path = os.path.join(folder_path, filename)
            dest_path = os.path.join(dest_dir, filename)

            if not os.path.exists(dest_path):
                # content of resize logic
                with Image.open(src_path) as img:
                    # Resize if larger than 640px on either side
                    max_dim = 640
                    if img.width > max_dim or img.height > max_dim:
                        ratio = min(max_dim / img.width, max_dim / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        print(f"Resized {filename} to {new_size}")
                    
                    img.save(dest_path)
                    print(f"Saved {filename} to training folder.")
            else:
                print(f"{filename} already in training folder.")

            print(f"Processing {filename}...")
            add_dataset_image(dest_path)

def dataset_mode():
    clear()
    command = input("""
s - select image
sf - select folder
c - clear dataset
e - exit
Enter command: """)
    if command == "s":
        filename = select_image()
        add_dataset_image(filename)
    elif command == "sf":
        add_folder_images()
    elif command == "c":
        confirmation = input("Are you sure you want to clear the dataset? (y/n) ")
        if confirmation == "y":
            clear_dataset()
        else:
            print("Cancelled")
    elif command == "e":
        exit()
    else:
        print("Invalid command")

def select_image():
    tk.Tk().withdraw()
    root = tk.Tk()
    root.attributes("-topmost", True) 
    root.withdraw()
    filename = askopenfilename(parent=root, initialdir="./dataset_images/train/") # File selection dialog
    root.destroy()
    return filename

def training_mode():
    # Set the number of epochs for training
    num_epochs = 10

    # Loop through each epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train the model for one epoch, printing status every 25 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training

        # Evaluate the model on the validation dataset
        evaluate(model, val_loader, device=device)  # Using val_loader for evaluation

        # Optionally, save the model checkpoint after each epoch
        torch.save(model.state_dict(), f"faster-rcnn-torch\model_epoch_{epoch + 1}.pth")

def test_mode():
    # class names
    label_list= ["","European Green","Native Rock","Native Jonah"]

    # Number of classes (include background)
    num_classes = 5   # The saved model was trained with 5 classes (likely due to auto-calculated classes + 1 background)

    # Load the same model architecture with correct number of classes
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

    """
    Change this to your model path
    """
    model.load_state_dict(torch.load(r"faster-rcnn-torch\model_epoch_10.pth"))
    model.eval()

    # Load image with OpenCV and convert to RGB
    img_path = r"dataset_images/train/egc.png" # CHANGE this to your image path
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform image
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).unsqueeze(0)

    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Parse predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Threshold
    threshold = 0.8
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = label_list[labels[i]]
            score = scores[i].item()

            # draw label and score
            text = f"{label}: {score:.2f}"
            cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw rectangle and label
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


    # Convert BGR to RGB for correct display with matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show image with larger figure size
    plt.figure(figsize=(16, 12))  # Increase size as needed
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = label_list[labels[i]]
            score = scores[i].item()
            print(f"Detected: {label} at position: {box}")

    if len(boxes) == 0:
        print("No crabs detected")

    time.sleep(30)


def main():
    
    while True:
        clear()
        mode = input(""" 
ds - dataset mode
tn - training mode
t - test mode
e - exit
Enter mode: """)
        if mode == "ds":
            dataset_mode()
        elif mode == "tn":
            training_mode()
        elif mode == "t":
            test_mode()
        elif mode == "e":
            exit()
        else:
            print("Invalid mode")

if __name__ == "__main__":
    main()
