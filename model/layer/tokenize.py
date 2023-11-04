import re
import os
import torch
import sys
sys.path.append('../clip')
from model import *
class Tokenize:
    def __init__(self, path):
        self.directory = path
        self.classes = []
    #get all classes
    def get_class_tokens(self):
        clip = 
        for filename in os.listdir(self.directory):
            if filename.endswith(".jpg"):  
                delimiters = '[.-]'
            result = re.split(delimiters, filename)
            result = [item for item in result if item]
            self.classes.append(result[2])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in range(len(self.classes)):
            self.classes[i] = clip.tokenize(self.classes[i]) #tokenize all classes
        text_inputs = torch.cat(classes).to(device)

        return self.classes


