import re, glob, os, pdb
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def load_accelerations(filename, dataset):
    accelerations_dict = {}
    with open(filename) as f:
        for line in f:
            hardcoded_path_video, a = line.split('\t')
            path_video = hardcoded_path_video
            accelerations_dict[path_video] = float(a)

    accelerations = []
    for d in dataset:
        path_video = d
        accelerations.append(accelerations_dict[path_video])
    return np.array(accelerations, dtype=np.float32)

def parse_angles(dataset):
    angles = []
    for d in dataset:
        path_video = d
        m = re.search(r'([12]0)_0[12]', path_video)
        rad_slope = float(m.group(1)) * np.pi / 180.
        angles.append(rad_slope)
    return np.array(angles, dtype=np.float32)

def get_initial_video_frame(path_video):
    path_img = path_video[:-4]
    path_img = path_img.replace('/', '-')
    path_img = os.path.join('frames', path_img + '.jpg')
    img = Image.open(path_img).convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    return img

def load_dataset(path_experiment, ramp_surface=1, size_batch=1, return_filenames=False):
    # List all video files
    path = os.path.join(path_experiment, "*", "*0_0{}".format(ramp_surface), "*", "Camera_1.mp4")
    fnames = glob.glob(path)
    fnames = [fname.replace(os.sep, "/") for fname in fnames]

    # Compute size of dataset
    size_dataset = len(fnames) - 1  # 1804 (w/o last video - broken)
    num_batches = int((size_dataset + 0.5) / size_batch)
    num_test = num_batches // 5  # 1
    num_train = num_batches - num_test  # 14

    # Load acceleration labels
    accelerations = load_accelerations('accelerations.log', fnames)
    parsed_angles = parse_angles(fnames)

    # Load first frame from each video and normalize
    images = [get_initial_video_frame(fname) for fname in fnames]
    images = torch.stack(images)
    angles = torch.tensor(parsed_angles, dtype=torch.float32)
    accelerations = torch.tensor(accelerations, dtype=torch.float32)

    if return_filenames:
        input_tuple = (images, angles, fnames)
    else:
        input_tuple = (images, angles)

    dataset = list(zip(*input_tuple, accelerations))  # Include filenames directly if required

    # Create train and test datasets
    test_indices = range(num_test)
    train_indices = range(num_test, len(dataset))
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=size_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=size_batch, shuffle=False)

    return train_loader, test_loader

