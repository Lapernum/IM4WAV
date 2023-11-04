import re
import os

directory = 'INSERT DIRECTORY PATH'
classes = []
#get all classes
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):  
      delimiters = '[.-]'
      result = re.split(delimiters, filename)
      result = [item for item in result if item]
      classes.append(result[2])
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) #load clip model


for i in range(len(classes)):
  classes[i] = clip.tokenize(classes[i]) #tokenize all classes
text_inputs = torch.cat(classes).to(device)

def image_CLIP(image, text, m):
  
  with torch.no_grad():
    image_features = m.encode_image(image)
    text_features = m.encode_text(text)
  image_features /= image_features.norm(dim=-1, keepdim=True)
  text_features /= text_features.norm(dim=-1, keepdim=True)
  similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
  values, indices = similarity[0].topk(1)
  return classes[indices[0]] # return predicted class
'''USE FUNCTION ON EACH IMAGE TO GET PREDICTED IMAGE
Like this:  
for each image:
  predicted_class = image_clip(image, text_inputs, model)
'''
