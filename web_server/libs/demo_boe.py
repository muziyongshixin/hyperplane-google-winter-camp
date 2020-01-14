import torch
import os, sys
import numpy as np
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000 
import torch.nn as nn
import torch.autograd as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from IPython import embed

root = u'/k/project/BOE'
img_dir   = os.path.join(root, u'images')
img_list_file = os.path.join(root, u'metadata', u'image_list.txt')
feature_file = os.path.join(root,u'misc',u'resnet50.npy')
num_workers = 8
batch_size = 1

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    
    def forward(self, x):
        return x

class imageDataset(Dataset):
    def __init__(self,img_dir,img_list,transform=None):
        self.img_dir = img_dir
        self.img_list = img_list #os.listdir(self.img_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,idx):
        img_name = self.img_list[idx]
        tmp_path = os.path.join(self.img_dir,img_name)
        image = Image.open(tmp_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def Get_Image_List(img_list_file):
    with open(img_list_file, 'r') as f:
        img_list = f.readlines()
    
    n_img = len(img_list)
    for i in range(0,n_img):
        img_list[i] = img_list[i].rstrip()
    
    return img_list

use_gpu = torch.cuda.is_available()
# resnet model
base_model = models.resnet50(pretrained = True)
#base_model.last_layer_name = 'fc'
base_model.fc = IdentityLayer()
#base_model.fc = nn.Linear(2048, 2048)
#torch.nn.init.eye(base_model.fc.weight)
base_model.eval()
if use_gpu:
    base_model = base_model.cuda()

img_transform = transforms.Compose([
        #transforms.Resize(256),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def Search_By_Image_Resnet50(query_image_filepath, transform=None):
    image = Image.open(query_image_filepath).convert('RGB')
    image = transform(image)
    return 

if __name__ == '__main__':
    print(sys.argv)
    assert len(sys.argv) == 2
    query_image_filepath = sys.argv[1]
    
    image = Image.open(query_image_filepath).convert('RGB')
    input = img_transform(image)
    
    #Search_By_Image_Resnet50(query_image_filepath, img_transform)
    img_list = Get_Image_List(img_list_file)
    n_img = len(img_list)
    res_list = []
    if use_gpu:
        print("use gpu")
        input = input.cuda()
    else:
        pass
    try:
        input.unsqueeze_(0)
        output = base_model(input).data.cpu().numpy()
        output = output.squeeze()
        #feature[cc:cc+batch_size,:] = output
        res_list = img_list[1:20]
    except:
        print('something weird happened!')
    print(res_list) 
