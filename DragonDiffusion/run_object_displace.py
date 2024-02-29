from PIL import Image
import numpy as np
from src.demo.model import DragonModels
import argparse
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = DragonModels(pretrained_model_path=pretrained_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Object Segmentation")
    parser.add_argument('--image',type=str) # Path to the image
    parser.add_argument('--class_name',type=str) # Name of object to segment
    parser.add_argument('--output',type=str) # Output path of image with segmented object
    parser.add_argument('--mask_path',type=str) # Path to store the object mask image
    parser.add_argument('--x',type=int) # Displacement along x axis
    parser.add_argument('--y',type=int) # Displacement along y axis

    args = parser.parse_args()
    
    # Loading image to be edited and its mask
    orig_image = np.array(Image.open(args.image))
    mask_img =   np.array(Image.open(args.mask_path))
    #ref_mask_img = np.array(Image.open("/data/rishubh/Raghav/GS/GD/segment-anything/images_seg/Backpack_seg_shift.png"))
    ref_mask_img = None#"/data/rishubh/Raghav/GS/GD/segment-anything/images_seg/Backpack_seg_shift.png"
    edited_image = model.run_move(original_image=orig_image, 
               mask=mask_img, 
               mask_ref=ref_mask_img, 
               prompt=args.class_name, 
               resize_scale=1, 
               w_edit=4, 
               w_content=6, 
               w_contrast=0.2, 
               w_inpaint=0.8, 
               seed=42, 
               selected_points=[[0,0],[args.x,-args.y]], 
               guidance_scale=4, 
               energy_scale=0.5, 
               max_resolution=1024, 
               SDE_strength=0.4, 
               ip_scale=0.1)
    
    # Saving Edited Image
    edited_image = np.array(edited_image[0])
    edited_image = Image.fromarray(edited_image)
    h, w, _ = orig_image.shape
    edited_image = edited_image.resize((h,w))
    edited_image.save(args.output)