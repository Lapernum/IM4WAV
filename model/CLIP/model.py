import torch
import clip

class Clip:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device) #load clip model
    def image_CLIP(image, text, m):
  
        with torch.no_grad():
            image_features = m.encode_image(image)
            text_features = m.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        return classes[indices[0]] # return predicted class