import re
import os


#get all classes
def get_all_classes(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  
            delimiters = '[.-]'
        result = re.split(delimiters, filename)
        result = [item for item in result if item]
        classes.append(result[2])
path = 'INSERT DIRECTORY PATH'
#classes = get_all_classes(path)


