from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--img_file', type=str, default='./datasets/afhq/', #
                            help='training and testing dataset')#15,139,17,396,3756, qinfeng /home/xdjf/qinfeng/data/celeba-1024/ ./datasets/celeba-hq/396/
        parser.add_argument('--no_shuffle', type=bool, default=True, help='if true, takes images serial')
        parser.add_argument('--mask_file', type=str, default='./datasets/mask/',
                            help='load test mask')  # ./datasets/mask/
        parser.add_argument('--mask_type', type=int, default=[4],
                            help='mask type, 0: center mask 1:random regular mask, '
                            '2: random irregular mask.3: random irregular mask2 4: external irregular mask. [0],[1,2],[1,2,3]')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, default=1, help='choice # reasonable results based on the discriminator score')

        self.isTrain = False

        return parser
