import urllib
import torch
import cv2
import re
import matplotlib
import numpy as np
from torchvision import transforms

from PIL import Image
from layer.depth import getDepthHead
from layer.segmentation import getSegmentationHead, getSegmentationModel
from mmseg.apis import inference_segmentor

import model.dinov2.eval.segmentation.utils.colormaps as colormaps

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

HEAD_DATASET = "voc2012"

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

DISTANCE_PERCENTAGE = 0.9

CLIP_THRESHOLD = 32

OUTPUT_PATH = ""


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


def make_segmentation_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ])


def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    # prevent index from being out of bound
    segmentation_logits[segmentation_logits + 1 >= colormap_array.shape[0]] = colormap_array.shape[0] - 2
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)


def tensor_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
def to_gray(input_image):
    """Convert a BGR image to grayscale.
    
    Args:
        input_image: cv2 numpy array (H, W, C) with C in BGR
 
    Returns:
        output_image: cv2 numpy array (H, W)
    """
    output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return output_image


def round_gray(input_image):
    """Round a numpy array image to 2 decimal places.
    
    Args:
        input_image: cv2 numpy array (H, W)
 
    Returns:
        output_image: cv2 numpy array (H, W)
    """
    output_image = np.around(input_image, decimals=2)
    return output_image


def open_image(path):
    """Open an image.
    
    Args:
        path: string path of the image
 
    Returns:
        output_image: PIL Image
    """
    output_image = Image.open(path)
    return output_image


def predict_depth(input_image):
    """Predict the depth information of an image.
    
    Args:
        input_image: PIL Image
 
    Returns:
        output_image: cv2 numpy array image (H, W, C) with the depth information
    """
    transform = make_depth_transform()

    scale_factor = 1
    rescaled_image = input_image.resize((scale_factor * input_image.width, scale_factor * input_image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    model = getDepthHead().cuda()

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    depth_image = render_depth(result.squeeze().cpu())
    tensor_transformer = tensor_transform()
    return np.array(tensor_transformer(depth_image)).transpose((1, 2, 0)) # Convert (C, H, W) to (H, W, C)


def predict_segmentation(input_image, model_choice):
    """Predict the segmentation information of an image.
    
    Args:
        input_image: PIL Image
        model_choice: String "head" (use pretrained head) or "model" (use mask2former pretrained model)
 
    Returns:
        output_image: cv2 numpy array image (H, W, C) with the segmentation information
    """
    transform = make_segmentation_transform()

    transformed_image = transform(input_image)

    if model_choice == "head":
        seg_model = getSegmentationHead()
    elif model_choice == "model":  
        seg_model = getSegmentationModel()
    else:
        print("Invalid model choice")
        return

    array = np.array(transformed_image)[:, :, ::-1] # BGR
    segmentation_logits = inference_segmentor(seg_model, array)[0]
    segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
    tensor_transformer = tensor_transform()
    return np.array(tensor_transformer(segmented_image)).transpose((1, 2, 0)) # Convert (C, H, W) to (H, W, C)


def clip_segmentation_with_volume(depth_image, segmentation_image, original_path):
    """Clip the image by segmentation and get the volumn factor for each segment.
    Will save the output images to the OUTPUT_PATH folder with filename as "video_index-image_index.ROI-segment_index.png"
    
    Args:
        depth_image: cv2 numpy array image (H, W, C)
        segmentation_image: cv2 numpy array image (H, W, C)
        original_path: string path of the original image
 
    Returns:
        volume_factor: a list of tuples, including volumn factors for each segment with Left and Right audio channel
    """
    gray_segmented_image = to_gray(segmentation_image)
    gray_depth_image = to_gray(depth_image)
    rounded_segmented_image = round_gray(gray_segmented_image)
    
    cv_original_image = cv2.imread(original_path)
    cv_original_image = cv2.resize(cv_original_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    ROI_number = 0
    volume_factor = []
    for gray_scale in list(np.array(range(101)) / 100.0):
        # Morph open to remove noise
        test_gray_image = np.copy(rounded_segmented_image)
        test_gray_image[test_gray_image!=gray_scale] = 0
        test_gray_image[test_gray_image==gray_scale] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        opening = cv2.morphologyEx(test_gray_image, cv2.MORPH_OPEN, kernel, iterations=1).astype('uint8')

        masked_depth_image = np.multiply(gray_depth_image, test_gray_image)

        # Find contours, obtain bounding box, extract and save ROI
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w < CLIP_THRESHOLD or h < CLIP_THRESHOLD:
                continue
            # c_mask = np.zeros((240, 320), np.uint8)
            # cv2.drawContours(c_mask, c, -1, (255, 255, 255), -1)
            ROI_depth = masked_depth_image[y:y+h, x:x+w]
            ROI_depth = ROI_depth[ROI_depth != 0]
            ROI_depth_mean = np.mean(ROI_depth)
            ROI_horizontal_factor = ((IMAGE_WIDTH / 2) - (x + 0.5 * w)) / float(IMAGE_WIDTH)
            volume_factor.append((ROI_depth_mean * DISTANCE_PERCENTAGE + ROI_horizontal_factor * (1 - DISTANCE_PERCENTAGE), ROI_depth_mean * DISTANCE_PERCENTAGE - ROI_horizontal_factor * (1 - DISTANCE_PERCENTAGE)))
            ROI = cv_original_image[y:y+h, x:x+w]
            
            delimiters = '[.]'
            result = re.split(delimiters, original_path)
            result = [item for item in result if item]
            
            cv2.imwrite(OUTPUT_PATH + '/' + result[0] + '.ROI-{}.png'.format(ROI_number), ROI)
            ROI_number += 1

    return volume_factor