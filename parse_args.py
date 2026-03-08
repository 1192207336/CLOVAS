import argparse
def test_parse_args():
    parser = argparse.ArgumentParser("CLOVAS", add_help=True)
    # paths
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--out_indice", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    # parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--config", type=str, default='./configs/CLOVAS.json', help="config file")
    parser.add_argument(
        '--vis', action='store_true', help='Use Flip and Multi scale aug')
    args = parser.parse_args()
    return args
class nbArgs:
    def __init__(self):
        self.checkpoint_path = './checkpoint/'
        self.dataset = 'mvtec'
        self.out_indice = [6, 12, 18, 24]
        self.image_size = 518
        self.sigma = 4
        self.config = './configs/CLOVAS.json'
        self.vis = False
def get_nb_args():
    args = nbArgs()
    return args
def train_parse_args():
    parser = argparse.ArgumentParser("CLOVAS", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=5, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--config", type=str, default='./configs/CLOVAS.json', help="config file")
    args = parser.parse_args()
    return args