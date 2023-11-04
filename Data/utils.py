import re
import os


#get all classes
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
#path = '../example_images'
#classes = get_all_classes(path)


