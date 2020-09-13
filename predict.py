import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import helper
from dataset import SimDataset
from resnet_unet import ResNetUNet

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

num_class = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetUNet(num_class).to(device)

model.load_state_dict(torch.load('checkpoints/best.pth', map_location=device))

model.eval()  # Set model to evaluate mode

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

test_dataset = SimDataset(3, transform=trans)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)
pred = torch.sigmoid(pred)
pred = pred.data.cpu().numpy()
print(pred.shape)

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [helper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

cv2.imshow('pred', pred_rgb[2])
cv2.waitKey(0)

# helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])