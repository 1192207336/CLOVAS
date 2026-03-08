import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import CLOVAS_lib
import torch
import argparse
import torch.nn.functional as F
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import random
from utils import get_transform
from config import DATA_ROOT
import json
import time
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def format_time(seconds):  
    hours = int(seconds // 3600)  
    minutes = int((seconds % 3600) // 60)  
    secs = int(seconds % 60)  
    return f'{hours:02}:{minutes:02}:{secs:02}'
def train(args):
    logger = get_logger(args.save_path)
    logger.info('config file: {}'.format(args.config))
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        model_configs = json.load(f)
    model_configs["dataset"]=args.dataset
    json_file = os.path.join(args.save_path,"model_config.json")
    with open(json_file,'w',encoding='utf-8') as f:
        json.dump(model_configs, f,ensure_ascii=False,indent=4)
    logger.info('model_configs: {}'.format(model_configs))
    model, _ = CLOVAS_lib.load("ViT-L/14@336px", device=device,configs = model_configs)
    model.train()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.to(device)
    model.visual.apply_CLIP_attention_surgery()
    ##########################################################################################
    base_lr = model_configs["learning_rate"]
    param_groups = [  
    {'params': model.visual.parameters(), 'lr': base_lr if "vpt_settings" in model_configs else 0},  
    {'params': model.oa_prompt_learner.parameters(), 'lr': base_lr},  
    {'params': model.text_encoder.parameters(), 'lr': base_lr},  
    {'params': model.decode_head.parameters(), 'lr': base_lr},  
    ]  
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01,betas=(0.5, 0.999))

    # optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=0.01,betas=(0.5, 0.999))

    times = []
    total_iters = args.epoch * len(train_dataloader) 

    best_loss = float('inf')  
    best_epoch = -1  
    best_weights = None  
    for epoch in range(args.epoch):
        loss_list = []
        for iter,items in enumerate(train_dataloader):
            iter_start_time = time.time()
            image = items['img'].to(device)
            labels =  items['anomaly'].to(device)
            masks = items['img_mask'].squeeze().to(device)
            gt = {
                'masks': masks,
                'labels': labels
            }

            losses = model.forward_train(image,gt)
            # print(f"losses是{losses}")
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time
            times.append(iter_time)  

            elapsed_time = sum(times)  
            avg_iter_time = np.mean(times)  
            remaining_iters = total_iters - (epoch * len(train_dataloader) + iter + 1)  
            remaining_time = remaining_iters * avg_iter_time 
            if (iter+1) % args.print_freq == 0:
                output_str = 'Epoch [{}/{}], Iter [{}/{}], eta: {}, '.format(epoch + 1, args.epoch,iter + 1, len(train_dataloader),format_time(remaining_time))
                for loss_name, loss_value in losses.items():  
                    output_str += '{}:{:.4f}, '.format(loss_name, loss_value) 
                output_str+= 'loss:{:.4f}'.format(loss.item())
                logger.info(output_str)
        avg_epoch_loss = np.mean(loss_list)  
        logger.info('Epoch [{}/{}], average_loss:{:.4f}'.format(epoch + 1, args.epoch, avg_epoch_loss))
        if avg_epoch_loss < best_loss:  
            best_loss = avg_epoch_loss  
            best_epoch = epoch + 1  
            best_weights = model.state_dict()  
            best_ckpt_path = os.path.join(args.save_path, 'best.pth')
            filtered_best_weights = {k: v for k, v in best_weights.items() if "prompt" in k or "decode_head" in k}  
            torch.save(filtered_best_weights, best_ckpt_path)  
            logger.info('Saving best model with lowest loss at epoch {} into {}'.format(best_epoch, best_ckpt_path)) 

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if "prompt" in k or "decode_head" in k} 
            torch.save(filtered_state_dict, ckp_path)
            logger.info('Saving checkpoints at {}-th epoch into {}'.format(epoch + 1,ckp_path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("CLOVAS", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/home/host/lcr/datasets/visa", help="train dataset path")
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
    
    args.dataset = "mvtec"
    args.train_data_path = DATA_ROOT[args.dataset]
    args.save_path = "" #
    args.config = r"./configs/CLOVAS.json" #CLOVAS.json
    print(f"checkpoints will be saved at {args.save_path}")
    setup_seed(args.seed)
    train(args)

    