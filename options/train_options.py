from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=5000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train',  default=False, help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_L1', type=float, default=100, help='wieght of the L1 loss')
        parser.add_argument('--val_freq', type=int, default=1, help='run validation every N epochs')
        parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
        parser.add_argument('--val_num_threads', type=int, default=1, help='validation data loader workers')
        parser.add_argument('--val_max_dataset_size', type=int, default=float("inf"), help='maximum validation pairs')
        parser.add_argument('--disable_val', action='store_true', help='disable validation and best-checkpoint saving')
        parser.add_argument('--curve_dpi', type=int, default=300, help='DPI for saved PNG training curves')
        self.isTrain = True
        return parser
