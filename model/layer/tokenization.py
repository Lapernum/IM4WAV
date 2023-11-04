import re
import os
import torch
import sys
sys.path.append('../clip')
from model import *
class Tokenize:

    def image_CLIP(image, text, m, classes):
        with torch.no_grad():
            image_features = m.encode_image(image)
            text_features = m.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        return classes[indices[0]] # return predicted class


