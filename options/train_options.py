from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--img_file', type=str, default='/media/sdc/qinfeng/afhq/', #
                            help='training and testing dataset')
        parser.add_argument('--mask_type', type=int, default=[4,5],
                            help='mask type, 0: center mask 1:random regular mask, '
                            '2: random irregular mask.3: random irregular mask2 4: external irregular mask. [0],[1,2],[1,2,3]')
        parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        parser.add_argument('--no_shuffle', type=bool, default=False, help='if true, takes images serial')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--mask_file', type=str, default='/media/sdc/qinfeng/mask/testing_mask_dataset/',
                            help='load test mask')  # test_mask/mask/testing_mask_dataset/
        # training epoch
        parser.add_argument('--phase', type=str, default='train', help='train, val, test')
        parser.add_argument('--iter_count', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--max_iteration', type=int, default=100, help='# of iter with initial learning rate')
        # learning rate and loss weight
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|exponent|cosine]')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=5000000, help='# of iter with initial learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to decay learning rate to zero')
        parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['wgangp', 'wgandiv', 'hinge', 'lsgan'])

        parser.add_argument('--SEED', type=int, default=10, help='random seeds')

        self.isTrain = True

        return parser
