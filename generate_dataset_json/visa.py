import os
import json
import pandas as pd
from prompts.visa_parameters import manual_prompts,gpt_prompts,normal_prompts
import re  
from llm_prompts_generator import llm_prompts_generate

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

id2class_map = {
        "candle": {"0": "BACKGROUND", "1": "Chunk Of Wax Missing", "2": "Damaged Corner of Packaging",
                "3": "weird candle wick", "4": "Different Colour Spot", "5": "Extra Wax in Candle",
                "6": "Foreign Particals on Candle", "7": "Wax Melded out of the candle", "8": "Other"},
        "capsules": {"0": "BACKGROUND", "1": "Bubble", "2": "Discolor", "3": "Scratch", "4": "Leak", "5": "Misshape"},
        "fryum": {"0": "BACKGROUND", "1": "Burnt", "2": "Corner or Edge Breakage", "3": "Different Colour Spot",
                "4": "Fryum Stuck Together", "5": "Middle Breakage", "6": "Similar Colour Spot", "7": "Small Scratches",
                "8": "Other"},
        "cashew": {"0": "BACKGROUND", "1": "Burnt", "2": "Corner or Edge Breakage", "3": "Different Colour Spot",
                "4": "Middle Breakage", "5": "Small Holes", "6": "Small Scratches", "7": "Stuck Together",
                "8": "Same Colour Spot", "9": "Other"},
        "chewinggum": {"0": "BACKGROUND", "1": "Chunk of gum missing", "2": "Corner Missing", "3": "Scratches",
                    "4": "Similar Colour Spot", "5": "Small Cracks", "6": "Other"},
        "macaroni1": {"0": "BACKGROUND", "1": "Chip Around Edge And Corner", "2": "Different Colour Spot",
                    "3": "Middle Breakage", "4": "Similar Colour Spot", "5": "Small Cracks", "6": "Small Scratches",
                    "7": "Other"},
        "macaroni2": {"0": "BACKGROUND", "1": "Breakage down the middle", "2": "Color spot similar to the Object",
                    "3": "Different Color spot", "4": "Small chip around edge", "5": "Small Cracks",
                    "6": "Small Scratches", "7": "Other"},
        "pcb1": {"0": "BACKGROUND", "1": "Bent", "2": "Melt", "3": "Scratch", "4": "Missing"},
        "pcb2": {"0": "BACKGROUND", "1": "Bent", "2": "Melt", "3": "Scratch", "4": "Missing"},
        "pcb3": {"0": "BACKGROUND", "1": "Bent", "2": "Melt", "3": "Scratch", "4": "Missing"},
        "pcb4": {"0": "BACKGROUND", "1": "Burnt", "2": "Scratch", "3": "Missing", "4": "Damage", "5": "Extra",
                "6": "Wrong Place", "7": "Dirt"},
        "pipe_fryum": {"0": "BACKGROUND", "1": "Burnt", "2": "Corner And Edge Breakage", "3": "Different Colour Spot",
                    "4": "Middle Breakage", "5": "Similar Colour Spot", "6": "Small Scratches", "7": "Stuck Together",
                    "8": "Small Cracks", "9": "Other"}
}

    
def get_class_names_unique(manual_prompts):
    CLSNAMES_UNIQUE={}
    for name, entries in manual_prompts.items():  
        for entry in entries:
            CLSNAMES_UNIQUE[name] = entry[1]
    return CLSNAMES_UNIQUE

LLM_PROMPT = {
    'ENG':'I am creating anomaly text prompts for an anomaly detection dataset. Given a defect type dictionary where the keys represent object names and the values are lists of strings, each representing a type of defect, you, as a prompt generation expert, are requested to generate a prompt dictionary in a style consistent with the CLIP pre-training phase based on the provided defect type dictionary. The structure of the dictionary should be: LLM_prompts={object_name:{ori_defect_type:defect:prompt,...},...}. Here is the defect type dictionary:...',
    'CN':'我正在为异常检测数据集制作异常文本提示词，给定一个缺陷类型字典，这个字典中，键表示对象类型名，值为一个字符串列表，其中的每一个字符串代表一种异常类型，你作为一个提示词生成专家，请你根据我提供给你的缺陷类型字典生成一个符合CLIP预训练阶段的文本提示词风格的提示词字典，这个字典的结构是：LLM_prompts={object_name:{ori_defect_type:defect:prompt,...},...}。以下是缺陷类型字典：...'
}

class VisASolver(object):
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]
    CLSNAMES_UNIQUE = {
        'candle':'candle', 'capsules':'capsules', 'cashew':'cashew', 'chewinggum':'chewinggum', 'fryum':'fryum',
        'macaroni1':'macaroni', 'macaroni2':'macaroni', 'pcb1':'pcb', 'pcb2':'pcb', 'pcb3':'pcb',
        'pcb4':'pcb', 'pipe_fryum':'fryum',
    } # get_class_names_unique(manual_prompts)

    def __init__(self, root='data/visa'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        info = {phase: {} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            image_anns = pd.read_csv(f'{self.root}/{cls_name}/image_anno.csv', header=0)
            path_to_label_dict = dict(zip(image_anns['image'], image_anns['label']))
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data['label'] == 'anomaly' else False
                    defect_types = path_to_label_dict[data['image']] if is_abnormal else []
                    

                    info_img = dict(
                        img_path=data['image'],
                        mask_path=data['mask'] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name='',
                        object_category = self.CLSNAMES_UNIQUE[cls_name],
                        # defect_ids = defect_ids,
                        defect_types = defect_types,
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
        self.generate_prompts()
        
    def generate_prompts(self):
        category_dict = self.get_category_dict()
        llm_prompts_generate(category_dict, out_root ="generate_dataset_json/prompts", dataset = 'visa', n=-1)
    def get_category_dict(self):
        category_dict = {}
        for obj,categories in id2class_map.items():
            category_dict[obj] = list(categories.values())
        print(f"category:{category_dict}")
        return category_dict
    def get_defect_type_dict(self):
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        all_defect_types={}
        for cls_name in self.CLSNAMES:
            image_anns = pd.read_csv(f'{self.root}/{cls_name}/image_anno.csv', header=0)
            path_to_label_dict = dict(zip(image_anns['image'], image_anns['label']))
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            if not self.CLSNAMES_UNIQUE[cls_name] in all_defect_types:
                all_defect_types[self.CLSNAMES_UNIQUE[cls_name]] = set()
            for phase in self.phases:
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data[2] == 'anomaly' else False
                    defect_type = path_to_label_dict[data[3]] if is_abnormal else ''
                    if is_abnormal:
                        defect_type = [defect.strip() for defect in defect_type.split(',') if defect.strip()]
                        for defect in defect_type:
                            all_defect_types[self.CLSNAMES_UNIQUE[cls_name]].add(defect)
        print(all_defect_types)
        return all_defect_types
import cv2
import numpy as np
if __name__ == '__main__':
    runner = VisASolver(root='/mnt/ssd/home/jcheng/datasets/visa')
    # runner.run()
    runner.generate_prompts()

    
