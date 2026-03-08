import cv2
import os
from utils import normalize
import numpy as np
import torch
from matplotlib import pyplot as plt
def visualizer(pathes, anomaly_map, img_size, save_path=None, cls_name=None,save=True):
    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        if save:
            save_vis = os.path.join(save_path, 'hotmap', cls_name[idx], cls)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            cv2.imwrite(os.path.join(save_vis, filename), vis)
        return vis

def visualizer_single(image_path,anomaly_map, img_size):
    vis = cv2.cvtColor(cv2.resize(cv2.imread(image_path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    mask = normalize(anomaly_map[0])
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    return vis

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)

    if isinstance(scoremap, torch.Tensor):
        scoremap = scoremap.detach().cpu().numpy()

    scoremap = np.squeeze(scoremap)

    if scoremap.ndim == 1:
        h = int(np.sqrt(scoremap.shape[0]))
        if h * h == scoremap.shape[0]:
            scoremap = scoremap.reshape((h, h))
        else:
            scoremap = scoremap.reshape(-1, 1)

    if scoremap.ndim == 3:
        scoremap = scoremap[:, :, 0] if scoremap.shape[2] == 1 else scoremap[0]
    elif scoremap.ndim > 3 or scoremap.ndim < 2:
        raise ValueError(f"Invalid scoremap shape after processing: {scoremap.shape}")

    scoremap = (scoremap - scoremap.min()) / (scoremap.ptp() + 1e-8)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)

    if scoremap.shape[:2] != np_image.shape[:2]:
        scoremap = cv2.resize(scoremap, (np_image.shape[1], np_image.shape[0]))

    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


PALETTE3 = [
        [0, 0, 0],
        [255, 255, 0],
        [255, 215, 0],
        [255, 165, 0],
        [205, 205, 0],
        [255, 255, 100],
        [200, 200, 50]
]

PALETTE2=[
    [0, 0, 0],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
           [255, 0, 255], [255, 165, 0], [0, 255, 255], [255, 20, 147], 
           [255, 0, 144], [0, 255, 127]

]
PALETTE = [[0,0,0],[0, 192, 64],[192, 0, 0], [128, 192, 192],
           [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
           [255, 0, 255], [255, 165, 0], [0, 255, 255], [255, 20, 147], 
           [255, 0, 144], [0, 255, 127], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
               [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
               [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
               [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
               [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
               [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
               [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
               [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
               [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
               [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
               [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
               [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
               [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
               [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
               [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
               [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
               [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
               [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
               [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
               [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
               [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
               [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
               [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
               [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
               [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
               [64, 192, 96], [64, 160, 64], [64, 64, 0]]

def draw_text_box(img, text, position, color=(255, 0, 0), font_scale=0.5, thickness=1):  
    font = cv2.FONT_HERSHEY_SIMPLEX  
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]  
    text_x, text_y = position
    text_x = max(text_x+10,0)
    text_y = min(text_y+18,img.shape[0]-int(text_size[1]/2))  
    box_coords = ((text_x, text_y-int(text_size[1]/2)-10), (text_x + text_size[0], text_y + int(text_size[1]/2)))  
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)  
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness) 
def apply_semantic_map(image,mask,class_names = None,alpha=0.5):
    # masked_img = masked_img * 0.5 + img_mask[0].cpu().numpy().transpose(1,2,0) * 255 * 0.5
    img = image.copy().astype(np.float64)   
    color_map = PALETTE2#plt.get_cmap('viridis')(np.linspace(0, 1, len(class_names)))[:, :3] * 255
    mask = mask.numpy()[0]     
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    for cls in range(1, len(color_map)):
        mask_rgb[mask == cls] = color_map[cls]

    background_mask = (mask == 0)[:, :, None]
    foreground_mask = (mask > 0)[:, :, None]

    background_blend = img * 0.5

    foreground_blend = cv2.addWeighted(img, alpha, mask_rgb, 1 - alpha, 0)

    masked_img = (
        background_blend * background_mask + 
        foreground_blend * foreground_mask
    ).astype(np.uint8)
    # masked_img = cv2.addWeighted(img, alpha, mask_rgb, 1-alpha, 0).astype(np.uint8)
    
    if class_names:
        unique_classes = np.unique(mask)  
        for cls in unique_classes:  
            if cls < len(class_names):
                cls_mask = (mask == cls)  
                y_coords, x_coords = np.where(cls_mask)  
                if len(x_coords) > 0 and len(y_coords) > 0:  
                    x_center = int(np.mean(x_coords))  
                    y_center = int(np.mean(y_coords))  
                    draw_text_box(masked_img, class_names[int(cls)], (x_center, y_center),thickness=2)  
    return masked_img
if __name__ == '__main__':
    from parse_args import get_nb_args
    from config import DATA_ROOT
    from dataset import global_defect_classes
    from utils import get_transform
    from PIL import Image
    import torch
    from matplotlib import pyplot as plt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = get_nb_args()
    args.dataset = 'visa'
    object_type = 'candle'
    args.data_path = DATA_ROOT[args.dataset]
    class_names = list(global_defect_classes[args.dataset].keys())
    img_path = os.path.join(args.data_path, object_type,'Data/Images','Anomaly','099.JPG')
    mask_path = os.path.join(args.data_path, object_type,'Data/Masks','Anomaly','099.png')
    preprocess, target_transform = get_transform(args)
    # load image
    image  = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)
    img_mask = np.array(Image.open(mask_path).convert("RGB"))  #img_mask = cv2.imread(os.path.join(self.root, mask_path))[:, :, 0] 
    img_mask = img_mask[:, :, 0]
    img_mask = Image.fromarray(img_mask) 
    img_mask = target_transform(img_mask)
    img_mask = torch.as_tensor(np.array(img_mask)).long()
    img_mask = img_mask.unsqueeze(0)
    img_mask[img_mask>0] = 1
    img_mask = img_mask.float()

    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(cv2.resize(raw_img, (args.image_size, args.image_size)), cv2.COLOR_BGR2RGB)
    masked_img = apply_semantic_map(raw_img,img_mask,class_names)
    plt.imsave('masked_img.png',masked_img)
    plt.show()