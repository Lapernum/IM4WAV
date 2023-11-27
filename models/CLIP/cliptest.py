import os
import re

def get_all_classes(directory):
    classes = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            filename = filename.replace('#', '')
            delimiters = '[.-]'
            result = re.split(delimiters, filename)
            result = [item for item in result if item]
            classes.append(result[2])
    return classes
    
all_classes = get_all_classes('./imagetest')
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(all_classes).to(device)
clip_outputs = []
for filename in os.listdir('./imagetest'):
  if filename.endswith(".jpg"):
    image = preprocess(Image.open('./imagetest/' + filename)).unsqueeze(0).to(device)
    with torch.no_grad():
      image_features = model.encode_image(image)
      text_features = model.encode_text(text)

      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      values, indices = similarity[0].topk(1)
      clip_outputs.append(all_classes[indices[0]])
print(clip_outputs)