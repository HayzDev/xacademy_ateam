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

def dataset_mode():
    clear()
    command = input("""
s - select image
e - exit
Enter command: """)
    if command == "s":
        select_image()
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
    
    for image in annotations["images"]:
        if image["file_name"] == filename:
            image_id = image["id"]
            print("Image already exists in dataset")
            break

    annotations = json.load(open(r"dataset_images/actual_stuff/_annotations.coco.json"))
    
    im = cv2.imread(filename)
    area = cv2.selectROI("Select area", im, fromCenter=False, showCrosshair=False)
    
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
