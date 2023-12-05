import sys
import os
modules_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.append(modules_dir)
from im2wav_utils import *
from Data.meta import ImageHear_paths
from models.hparams import CLIP_VERSION
import clip
from tqdm import tqdm
from models.image_model import predict_depth, predict_segmentation, clip_segmentation_with_volume

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
    parser.add_argument("-path_list", dest='path_list', action='store', type=str)
    v = vars(parser.parse_args())
    print(v)
    return v

def getClass(dir_path):
    class_list = {}
    filenames = os.listdir(dir_path)
    for filename in filenames:
        filename_split = filename.split("-")
        curr_class = "-".join(filename_split[:-1])
        if curr_class not in class_list:
            class_list[curr_class] = [filename]
        else:
            class_list[curr_class].append(filename)
    return class_list

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
        if args['path_list'] is None:
            object2paths = ImageHear_paths
        else:
            object2paths = getClass(args['path_list'])
        image_objects = list(object2paths.keys())
        for i, object in enumerate(image_objects):
            object_files = object2paths[object]
            CLIP["image"][object] = {}
            for file in object_files:
                curr_image = Image.open(os.path.join(args['path_list'], file))
                depth_output = predict_depth(curr_image)
                segmented_output = predict_segmentation(curr_image, "head")
                volume_factors, cropped_images = clip_segmentation_with_volume(depth_output, segmented_output, os.path.join(args['path_list'], file))
                cropped_images = torch.cat([preprocess(image).unsqueeze(0).to(device) for image in cropped_images])
                image_features = model.encode_image(cropped_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()
                print(image_features.shape)
                print("image features class: ", object, file, image_features.shape)
                CLIP["image"][object][file] = {}
                CLIP["image"][object][file]["image_features"] = image_features
                CLIP["image"][object][file]["volume_factor"] = volume_factors

    with open(f"{args['save_dir']}/CLIP.pickle", 'wb') as handle:
        pickle.dump(CLIP, handle, protocol=pickle.HIGHEST_PROTOCOL)