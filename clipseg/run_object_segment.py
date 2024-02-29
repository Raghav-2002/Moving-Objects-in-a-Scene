import torch
import requests
from models.clipseg import CLIPDensePredT
from PIL import Image, ImageDraw
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import argparse

def save_mask(masks,mask_path):
    image_size = input_image.size
    black_image = Image.new("RGB", image_size, color="black")


    for col in range(0,len(masks)):
        for row in range(0,len(masks[0])):
            if masks[col][row]:
                black_image.putpixel((row, col), (255, 255, 255))

    black_image.save(mask_path)


def show_mask(mask, ax):
    color = np.array([255/255, 30/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Object Segmentation")
    parser.add_argument('--image',type=str) # Path to the image
    parser.add_argument('--class_name',type=str) # Name of object to segment
    parser.add_argument('--output',type=str) # Output path of image with segmented object
    parser.add_argument('--mask_path',type=str) # Path to store the object mask image
    parser.add_argument('--sam_ckpt',type=str) # Path to SAM checkpoints
    parser.add_argument('--clipdense_ckpt',type=str) # PAth to CLIPDenase checkpoint

    args = parser.parse_args()

    # Initializing variables
    image_path = args.image
    prompts = args.class_name
    sam_checkpoint = args.sam_ckpt
    model_type = "vit_h"
    device = "cuda"
    Resize_size = 352

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Loading SAM Model....")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    print("Loading CLIPDense Model...")
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load(args.clipdense_ckpt, map_location=torch.device('cpu')), strict=False)
    
    print("Segmenting the object .....")
    input_image = Image.open(image_path)
    width,height = input_image.size

    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((Resize_size, Resize_size)),])
    
    img = transform(input_image).unsqueeze(0)
    
    with torch.no_grad():
        preds = model(img, prompts)[0]

    threshold = 0.99
    
    input_point = []
    input_label = []

    flag = 0
    normalized_preds = (preds - torch.min(preds))/(torch.max(preds)-torch.min(preds))

    # Finding the co-ordinates on one point with high confidence score (i.e greater than threshold)
    for col in range(0,len(img[0][0])):
        for row in range(0,len(img[0][0][0])):
            if normalized_preds[0][0][row][col] > threshold:
                input_label.append(1) # 1 indicates foreground object for SAM 
                scaled_row = int(row*(width/Resize_size))
                scaled_col = int(col*(height/Resize_size))
                input_point.append([scaled_col,scaled_row])
                flag = 1
                break
        if flag == 1:
            break

    input_point = np.array(input_point)
    input_label = np.array(input_label)

    # Generating multiple mask containing the high confidence score point
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
      
    # Finding the best segementation from given set of masks, by maximizing segementation area
    index = 0
    max_segment_area = 0
    max_index = 0
    for mask in masks:
        mask.sum()
        segment_area = np.count_nonzero(mask)
        if segment_area > max_segment_area:
            segment_area = max_segment_area
            max_index = index
        index = index + 1

    # Saving segmented image and image mask 
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks[max_index], plt.gca())
    plt.axis('off')
    plt.savefig(args.output)

    print("Saving Mask...")
    save_mask(masks[max_index],args.mask_path)