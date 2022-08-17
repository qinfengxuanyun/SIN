import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from dataloader.data_loader import dataloader
from model import create_model
import os
from model.network import *
import numpy as np
import random

from tensorboardX import SummaryWriter
from util.util import dict_slice,tensor2im
from PIL import Image
from itertools import islice
import collections
#from mtcnn_pytorch.src import detect_face

if __name__ == '__main__':
    opt_test = TestOptions().parse()
    torch.backends.cudnn.deterministic = True  # cudnn

    model = create_model(opt_test)
    test_dataset = dataloader(opt_test)
    test_dataset_size = len(test_dataset) #* opt.batchSize
    print('test images = %d' % test_dataset_size)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(islice(test_dataset, opt_test.how_many)):
            model.set_input(data)
            model.test()