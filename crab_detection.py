"""Detects crabs in images"""

import json
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

device = torch.device("cpu")

def clear():
    os.system("cls")

def add_dataset_image(filename):
    annotations = json.load(open(r"dataset_images/actual_stuff/_annotations.coco.json"))
    
    for image in annotations["images"]:
        if image["file_name"] == filename:
            print("Image already exists in dataset")
            break
    
    category = input("Enter category: ")

    im = cv2.imread(filename)
    area = cv2.selectROI("Select area", im, fromCenter=False, showCrosshair=False)

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

    json.dump(annotations, open(r"dataset_images/actual_stuff/_annotations.coco.json", "w"))

    print("Success!")

def clear_dataset():
    annotations = json.load(open(r"dataset_images/actual_stuff/_annotations.coco.json"))
    annotations["images"] = []
    annotations["annotations"] = []
    json.dump(annotations, open(r"dataset_images/actual_stuff/_annotations.coco.json", "w"))

def dataset_mode():
    clear()
    command = input("""
s - select image
c - clear dataset
e - exit
Enter command: """)
    if command == "s":
        filename = select_image()
        add_dataset_image(filename)
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
    filename = askopenfilename(parent=root, initialdir="./dataset_images/actual_stuff/") # File selection dialog
    root.destroy()
    return filename
    
    
def main():
    clear()
    mode = input(""" 
ds - dataset mode
e - exit
Enter mode: """)
    if mode == "ds":
        dataset_mode()
    elif mode == "e":
        exit()
    else:
        print("Invalid mode")

if __name__ == "__main__":
    main()
