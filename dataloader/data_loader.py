from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset,make_dataset2
from util import task
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
#from util import faceflip
from skimage.feature import canny
from skimage.color import rgb2gray
import torchvision.transforms.functional as tf
#import matplotlib.pyplot as plt
# from prefetch_generator import BackgroundGenerator
class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        # self.img_paths, self.img_size = make_dataset(opt.img_file)
        self.mask_file = opt.mask_file
        self.img_paths,self.img_size = make_dataset(opt.img_file,opt.phase)
        # provides random file for training and testing
        if opt.mask_file != 'none':
            self.mask_paths, self.mask_size = make_dataset2(opt.mask_file)
        #print("mask_size",self.mask_size)
        self.transform = get_transform(opt)
        self.sigma = 2

    def __getitem__(self, index):
        # load image
        img, img_path = self.load_img(index)
        #img2, img_path2 = self.load_img2("train/cat/flickr_cat_000033.jpg") #flickr_cat_000033.jpg flickr_cat_000014.jpg
        # if self.opt.isTrain:
        #     landmark_path = self.opt.landmark_file + img_path.split('/')[-1]
        # else:
        #     landmark_path = self.opt.landmark_file2 + img_path.split('/')[-1]
        # img_pil = Image.open(landmark_path).convert('RGB')
        # landmark1 = self.transform(img_pil)
        # if self.opt.isTrain:
        #     img, landmark1 = my_transform2(img, landmark1)
        # else:
        #     img = tf.to_tensor(img)
        #     landmark1 = tf.to_tensor(landmark1)
        
        # load mask
        mask = self.load_mask(img, index)
        #kernel = np.ones((self.opt.fineSize[0] // 10, self.opt.fineSize[0] // 10), np.uint8)
        #dilate = cv2.dilate(mask.permute([1, 2, 0]).numpy(), kernel) #膨胀
        # cv2.imshow("img", mask.permute([1, 2, 0]).numpy())
        # cv2.imshow("img2", dilate)
        # cv2.waitKey(0)
        #dilate_mask = torch.from_numpy(dilate).permute([2,0,1])
        # edge1=self.load_edge(img)
        # if self.opt.isTrain:
        #     style_grad =  np.load(self.style_path+(img_path.split('/')[-1]).replace('jpg','npy')).reshape((512,1024))
        # else:
        #     style_grad =  np.ones((512,1024))
        return {'img': img, 'img_path': img_path,'mask': mask}#, 'style_grad': style_grad}#,'dilate_mask':dilate_mask}
        #return {'img': img, 'img_path': img_path, 'img2': img2, 'img_path2': img_path2,'mask': mask}#, 'style_grad': style_grad}#,'dilate_mask':dilate_mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.opt.img_file + self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)

        return img, img_path

    def load_img2(self, path):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.opt.img_file + path
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        return img, img_path
    
    def load_mask(self, img, index):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
        mask_type = self.opt.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        # random irregular mask2
        if mask_type == 3:
            return task.random_irregular_mask2(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 4:
            while(1):
                if self.opt.isTrain:
                    mask_index = random.randint(0, self.mask_size - 1)
                else:
                    mask_index = index%self.mask_size#random.randint(0, self.mask_size - 1)
                mask_pil = Image.open(self.mask_file + self.mask_paths[mask_index]).convert('RGB')
                size = mask_pil.size[0]
                if size > mask_pil.size[1]:
                    size = mask_pil.size[1]
                if self.opt.isTrain:
                    mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                         transforms.RandomRotation(10),
                                                         transforms.CenterCrop([size, size]),
                                                         transforms.Resize(self.opt.fineSize),
                                                         transforms.ToTensor()
                                                         ])
                else:
                    mask_transform = transforms.Compose([transforms.CenterCrop([size, size]),
                                                         transforms.Resize(self.opt.fineSize),
                                                         transforms.ToTensor()
                                                         ])
                mask = 1.0-(mask_transform(mask_pil) == 1).float()
                #mask = (mask_transform(mask_pil) == 1).float()
                mask_pil.close()

                if self.opt.isTrain:
                    temp = np.mean(1.0 - mask.numpy().transpose(1, 2, 0))
                    if temp < 0.5:
                        return mask
                else:
                    # temp = np.mean(1.0-mask.numpy().transpose(1, 2, 0))
                    # if temp > 0.2 and temp < 0.3:
                    #     return mask
                    return mask

        if mask_type == 5:
            return task.bbox2mask(img)
            #
            #return task.create_mask(img)

    def load_edge(self, img):
        edge = canny(rgb2gray(np.array(img).transpose(1, 2, 0)), sigma=self.sigma)
        # edge = np.expand_dims(edge, axis=2).astype(np.float32)
        # cv2.imshow("edge",edge)
        # cv2.waitKey(0)
        edge = np.expand_dims(edge, axis=0).astype(np.float32)
        edge = torch.from_numpy(edge)
        return edge

def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle,
                              num_workers=int(opt.nThreads))

    return dataset

def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    # if opt.isTrain:
    #     if opt.resize_or_crop == 'resize_and_crop':
    #         transform_list.append(transforms.Resize(osize))
    #         transform_list.append(transforms.RandomCrop(fsize))
    #     elif opt.resize_or_crop == 'crop':
    #         transform_list.append(transforms.RandomCrop(fsize))
    #     if not opt.no_augment:
    #         transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
    #     if not opt.no_flip:
    #         transform_list.append(transforms.RandomHorizontalFlip())
    #     if not opt.no_rotation:
    #         transform_list.append(transforms.RandomRotation(3))
    # else:
    #     transform_list.append(transforms.Resize(fsize))
    transform_list.append(transforms.Resize(fsize))
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

def my_transform2(image, landmark):
    angle = transforms.RandomRotation.get_params([-3, 3])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    landmark = landmark.rotate(angle)
    # 50%的概率应用水平翻转
    if random.random() > 0.5:
        image = tf.hflip(image)
        landmark = tf.hflip(landmark)
    image = tf.to_tensor(image)
    landmark = tf.to_tensor(landmark)
    return image, landmark
