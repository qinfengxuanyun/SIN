import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image,ImageDraw
import random
import math
###################################################################
# random mask generation
###################################################################

'''
def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask
'''
def create_mask(img):
    _,H,W = img.size()
    #H = H // 2
    #W = W // 2
    mask = np.ones([3,H,W]).astype(np.float32)
    mask_x = random.randint(0, W - W//2)
    mask_y = random.randint(0, H - H//2)
    mask[:,mask_y:mask_y + H//2, mask_x:mask_x + W//2] = 0
    '''
    s = img.size()
    mask_out = np.ones(s)
    scale = 4
    x = int(s[1] / scale)
    y = int(s[2] / scale)
    range_x = int(s[1] * (1 - 1 / scale))
    range_y = int(s[2] * (1 - 1 / scale))
    mask_out[:, x:range_x, y:range_y] = mask
    '''
    mask_out = torch.from_numpy(mask)

    return mask_out

def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    scale = 4
    x = int(size[1] / scale)
    y = int(size[2] / scale)
    range_x = int(size[1] * (1 - 1 / scale))
    range_y = int(size[2] * (1 - 1 / scale))
    mask[:, x:range_x, y:range_y] = 0

    return mask

def random_irregular_mask(img):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 10
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    number = random.randint(16, 32)#
    for _ in range(number):
        model = random.random()
        if model < 0.6:
            # Draw random lines
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            thickness = randint(2, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        elif model > 0.6 and model < 0.8:
            # Draw random circles
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            radius = randint(2, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        elif model > 0.8:
            # Draw random ellipses
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(2, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask
def random_regular_mask(img):
    """Generates a random regular hole"""
    while(1):
        mask = torch.ones_like(img)
        s = img.size()
        N_mask = random.randint(1, 3)
        limx = s[1] - s[1] / (N_mask + 1)
        limy = s[2] - s[2] / (N_mask + 1)
        for _ in range(N_mask):
            x = random.randint(0, int(limx))
            y = random.randint(0, int(limy))
            range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
            range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
            mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
        if 0.2<torch.mean(1.0-mask)<0.4:
            return mask
def random_irregular_mask2(img):
    while(1):
        size = img.size()
        _,H,W = img.size()
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 13
        max_width = 40

        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = 1.0-np.reshape(mask, (1, H, W))
        mask = np.concatenate([mask,mask,mask],axis=0)
        mask = torch.from_numpy(mask)
        if 0.1 < torch.mean(1.0 - mask) < 0.4:
            return mask
    #return mask

def random_bbox():
    img_shape = [256, 256, 3]
    img_height = img_shape[0]
    img_width = img_shape[1]
    height = 128
    width = 128
    maxt = img_height - height
    maxl = img_width- width
    t = np.random.randint(0, maxt)
    l = np.random.randint(0,maxl)
    h = height
    w = width
    return (t, l, h, w)
def npmask(bbox, height, width, delta_h, delta_w):
    mask = np.ones((3, height, width), np.float32)
    h = np.random.randint(delta_h//2+1)
    w = np.random.randint(delta_w//2+1)
    mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
         bbox[1]+w:bbox[1]+bbox[3]-w] = 0
    return mask
def bbox2mask(img):
    bbox = random_bbox()
    max_delta_height =32
    max_delta_width =32
    img_shape = [256, 256, 3]
    height = img_shape[0]
    width = img_shape[1]
    mask = npmask(bbox, height, width,max_delta_height, max_delta_width)
    mask = torch.from_numpy(mask)
    return mask

###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    #scaled_img = (scaled_img >= 0.5).float()
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs

def scale_up_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h * ratio
        nw = w * ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs