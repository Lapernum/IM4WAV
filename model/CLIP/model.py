import torch
import clip

class Clip:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device) #load clip model
        self.model = model
        self.preprocess = preprocess
    