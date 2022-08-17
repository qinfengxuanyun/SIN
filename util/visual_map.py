from pylab import *
import numpy as np
import cv2

dir = "./feature_map/"
dir2 = "./results2/"
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(map,name):
    image_numpy = map.detach().cpu().float().numpy()
    img_batch = (np.transpose(image_numpy, (0,2, 3, 1)))#*10
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        #title('feature_map_{}'.format(i))

    plt.savefig(dir+'feature_map'+name+'.png', dpi=200)

    plt.figure()
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig(dir+'feature_map_sum'+name+'.png', dpi=200)

def visualize_feature_map2(img_batch,name):
    print(img_batch.shape)
    cv2.imwrite(dir2+name+".png",np.uint8(img_batch*255))

    
