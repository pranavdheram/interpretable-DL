"""
The following script generates saliency maps based on methods listed in the file 'all_techniques.py. At its core, we have a image classification task. We use pretrained eural models on ImaegNet dataset which allows us to directly use saliency techniques without building models from scratch. The Readme provided in the directory explains how the code can be run.
Comments are provided in the code wherever we think necessary. 

Author: Pranav Dheram
"""
# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import sys
import matplotlib.image as mpimg
import pdb
import glob
import xml.etree.ElementTree as ET
import bounding_boxes as bounding
import os
from misc_functions import get_example_params, save_class_activation_images, convert_to_grayscale, save_gradient_images, get_positive_negative_saliency
import cam as cam_map
import all_techniques

# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
elif model_id == 4:
    net = models.alexnet(pretrained=True)
    finalconv_name = 'features'

net.eval()
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


def is_metadata_image(filename):
    try:
        image = Image.open(filename)
        return 'exif' in image.info
    except OSError:
        return False

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

#response = requests.get(IMG_URL)
#img_pil = Image.open(io.BytesIO(response.content))
#img_pil.save('test.jpg')
images = []
bounding_boxes = []
images_names = []
root = '/home/pdheram/data/Imagenet_data/'
for foldername in os.listdir(root):
    if foldername=='french_bulldog': continue
    class_folder = root+foldername
    image_folder = class_folder+'/images/'
    boxes_folder = class_folder+'/boxes/'
    for filename in glob.glob(image_folder+'*.jpg'):
        image_name = filename.split('/')[-1].split('.')[0]
        xml_file = boxes_folder+image_name+'.xml'
        if os.path.isfile(xml_file):
            try:
                img = Image.open(filename)
                img.verify()
                images.append(Image.open(filename))
                bounding_boxes.append(bounding.retrieve_bounding_box(xml_file))
                images_names.append(class_folder+'/cam_dark/'+image_name)
            except Exception as E:
                continue
#img_pil = Image.open(str(sys.argv[1]))
#img_tensor = preprocess(img_pil)
#img_variable = Variable(img_tensor.unsqueeze(0))
#logit = net(img_variable)

# download the imagenet category list
classes = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

for j, image in enumerate(images):
    #image = Image.open('test.jpg')
    img_tensor = preprocess(image)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if img_variable.shape[1]!=3: continue
    # predict class of the image, we'll later use this to generate saliency maps for that class
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
   # for i in range(0, 5):
   #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate saliency map for the top1 prediction
    CAMs = all_techniques.returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    cam = CAMs[0]
    #grad_cam = all_techniques.GradCam(net, target_layer=11)
    #cam = grad_cam.generate_cam(img_variable, idx[0])
    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    
    width, height = image.size
    heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
    #image = cv2.imread('test.jpg')
    
    result = heatmap * 0.5 + np.array(image) * 0.3
    #numpy_horizontal_concat = np.concatenate(
    # draw bounding box
    b_boxes = bounding_boxes[j]
    for box in b_boxes:
        cv2.rectangle(result, box[0], box[1], (0,255,0), 5)
    numpy_horizontal = np.hstack((image, heatmap, result))
    output_file = images_names[j]+'.jpg'
    cv2.imwrite(output_file, numpy_horizontal)
    #cv2.imwrite('lol.jpg', numpy_horizontal)
    #pdb.set_trace()
