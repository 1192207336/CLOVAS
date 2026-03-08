import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import CLOVAS_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_learners.AnomalyCLIP_prompt_learner import AnomalyCLIP_PromptLearner
from prompt_learners.RPG import RNN_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger,print_log
from tqdm import tqdm
from config import DATA_ROOT
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
import cv2
import json
from prettytable import PrettyTable
import time
from collections import OrderedDict
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics,intersect_and_union,pre_eval_to_metrics,image_level_metrics_single,pixel_level_metrics_single
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from PIL import Image
def generate_prompts(object_type):
    prompts,num_good_cls = None,1
    return prompts,num_good_cls
def get_test_time_prompts(object_type,dataset='visa'):
    prompts,num_good_cls = generate_prompts(object_type)
    return prompts,num_good_cls 
def test(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = os.path.join(os.path.dirname(args.checkpoint_path),"model_config.json")
    if not os.path.isfile(config_file):
        config_file = args.config
    config_file = args.config
    with open(config_file, "r") as f:
        model_configs = json.load(f)
    model_configs["dataset"] = args.dataset
    model_configs["training"] = False
    model_configs["cocoop_mode"] = False
    logger.info('Load checkpoint from: {}'.format(args.checkpoint_path))
    logger.info('model_configs: {}'.format(model_configs))
    model, _ = CLOVAS_lib.load("ViT-L/14@336px", device=device,configs = model_configs)

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint,False)
    model.visual.apply_CLIP_attention_surgery()

    model.to(device)
    model.eval()
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0]=1
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            out  = model.forward_test(image,topk=0.3)
            text_probs = out['text_probs']
            anomaly_map = out['anomaly_map']
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            if args.vis:
                print("Start saving！")
                visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)
        # if idx>500:
        #     break
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    print("Start to calculate metrics")

    for obj in tqdm(obj_list):
        table = []
        table.append(obj)
        if len(results[obj]['gt_sp'])==0:
            continue
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results)


def test_mIoU(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        model_configs = json.load(f)
    logger.info('model_configs: {}'.format(model_configs))
    model_configs["dataset"] = args.dataset
    model_configs["training"] = False
    model, _ = CLOVAS_lib.load("ViT-L/14@336px", device=device,configs = model_configs)

    # model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    

    checkpoint = torch.load(args.checkpoint_path)
    load_keys = set()
    for key in list(checkpoint.keys()):
        load_keys.add(key.split('.')[0])
    logger.info('load keys: {}'.format(load_keys))
    model.load_state_dict(checkpoint,False)

    # model.to(device)
    model.visual.apply_CLIP_attention_surgery()

    model.to(device)
    class_names = model.prompt_generator.class_names
    # class_names.pop(0)
    num_classes = len(class_names)
    
    results = []
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        gt_mask = items['img_mask'].squeeze(0)
        is_anomaly = items['anomaly'].detach().cpu()
        with torch.no_grad():
            if is_anomaly:
                out  = model.forward_test(image,topk=0.3,fuse_thresh=0.5)
                pred_masks = out['pred_masks'].squeeze(0).cpu()
                results.extend([intersect_and_union(
                        pred_masks,
                        gt_mask,
                        num_classes,
                        255,
                        label_map=dict(),
                        reduce_zero_label=False)])
        # if idx>100:
        #     break

    ret_metrics = pre_eval_to_metrics(results, 'mIoU')
    ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)
    print('+++++++++++ Total classes +++++++++++++')
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)
    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
            pAcc = val
        else:
            summary_table_data.add_column('m' + key, [val])
    print('per class results:')
    print(class_table_data.get_string())
    print('Summary:')
    print(summary_table_data.get_string())
    eval_results = {}
    # each metric dict
    for key, value in ret_metrics_summary.items():
        if key == 'aAcc':
            eval_results[key] = value / 100.0
        else:
            eval_results['m' + key] = value / 100.0

    ret_metrics_class.pop('Class', None)
    for key, value in ret_metrics_class.items():
        eval_results.update({
            key + '.' + str(name): value[idx] / 100.0
            for idx, name in enumerate(class_names)
        })
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = os.path.join(args.save_path,
                                 f'eval_mIoU_{timestamp}.json')
    with open(json_file,'w',encoding='utf-8') as f:
        json.dump(eval_results, f,ensure_ascii=False,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CLOVAS", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--config", type=str, default='./configs/CLOVAS.json', help="config file")
    parser.add_argument(
        '--vis', action='store_true', help='Use Flip and Multi scale aug')
    args = parser.parse_args()
    # print(args)
    # args.dataset = 'visa'
    args.data_path = DATA_ROOT[args.dataset]
    args.checkpoint_path = r''
    args.config = r"./configs/CLOVAS.json"
    # args.save_path = os.path.join(os.path.dirname(args.checkpoint_path), 'eval_results')
    # args.vis = False
    args.vis = False
    setup_seed(args.seed)
    test(args)

