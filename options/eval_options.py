from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--img_file', type=str, default='/home/xdjf/qinfeng/data/celeba-1024/',
                            help='training and testing dataset')#15,139,17,396,3756, ./datasets/celeba-hq/396/ /home/xdjf/qinfeng/data/celeba-1024/
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./eval/', help='saves results here')
        parser.add_argument('--how_many', type=int, default=10, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, default=1, help='choice # reasonable results based on the discriminator score')

        self.isTrain = False

        return parser
