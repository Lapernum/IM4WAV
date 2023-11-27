import sys
import os
modules_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.append(modules_dir)
from im2wav_utils import *
from Data.meta import ImageHear_paths
from models.hparams import CLIP_VERSION
import clip
from tqdm import tqdm

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, str])
    """
    import argparse
    parser = argparse.ArgumentParser(description='collect CLIP')
    parser.add_argument("-save_dir", dest='save_dir', action='store', type=str, default="image_CLIP")
    parser.add_argument("-path_dir", dest='path_dir', action='store', type=str)
    v = vars(parser.parse_args())
    print(v)
    return v

def get_all_classes(directory):
    file_classes = {}
    print("Getting all classes")
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jpg"):
            filename_hyphen = filename.split("-")
            fileclass = filename_hyphen[0]
            if fileclass not in file_classes:
                file_classes[fileclass] = [filename]
            else:
                file_classes[fileclass].append(filename)
    return file_classes

if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLIP = {}
    CLIP["image"] = {}
    with torch.no_grad():
        # generate images CLIP
        model, preprocess = clip.load(CLIP_VERSION, device=device)
        image_features = []
        if args['path_dir'] is None:
            print("Please provide a path list to the images")
            sys.exit()
        else:
            object2paths = get_all_classes(args['path_dir'])
        image_objects = list(object2paths.keys())
        for i, object in enumerate(image_objects):
            images = torch.cat([preprocess(Image.open(args['path_dir'] + "/" + path)).unsqueeze(0).to(device) for path in object2paths[object]])
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
            print("image features class: ", object, image_features.shape)
            CLIP["image"][object] = image_features

    with open(f"{args['save_dir']}/CLIP.pickle", 'wb') as handle:
        pickle.dump(CLIP, handle, protocol=pickle.HIGHEST_PROTOCOL)