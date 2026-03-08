import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import cv2

local2global_id_map = {
    "visa": {
        "candle": { "BACKGROUND": 0, "Chunk Of Wax Missing": 1, "Damaged Corner of Packaging": 1, "weird candle wick": 4, "Different Colour Spot": 2, "Extra Wax in Candle": 4, "Foreign Particals on Candle": 4, "Wax Melded out of the candle": 3, "Other": 6 }, 
        "capsules": { "BACKGROUND": 0, "Bubble": 4, "Discolor": 2, "Scratch": 2, "Leak": 3, "Misshape": 1 }, 
        "fryum": { "BACKGROUND": 0, "Burnt": 3, "Corner or Edge Breakage": 1, "Different Colour Spot": 2, 
                  "Fryum Stuck Together": 4, "Middle Breakage": 1, "Similar Colour Spot": 2, "Small Scratches": 2, "Other": 6 }, 
        "cashew": { "BACKGROUND": 0, "Burnt": 3, "Corner or Edge Breakage": 1, "Different Colour Spot": 2, 
                   "Middle Breakage": 1, "Small Holes": 2, "Small Scratches": 2, "Stuck Together": 4, "Same Colour Spot": 2, "Other": 6 }, 
        "chewinggum": { "BACKGROUND": 0, "Chunk of gum missing": 1, "Corner Missing": 1, "Scratches": 2, 
                       "Similar Colour Spot": 2, "Small Cracks": 2, "Other": 6 }, 
        "macaroni1": { "BACKGROUND": 0, "Chip Around Edge And Corner": 1, "Different Colour Spot": 2, 
                      "Middle Breakage": 1, "Similar Colour Spot": 2, "Small Cracks": 2, "Small Scratches": 2, "Other": 6 }, 
        "macaroni2": { "BACKGROUND": 0, "Breakage down the middle": 1, "Color spot similar to the Object": 2, 
                      "Different Color spot": 2, "Small chip around edge": 1, "Small Cracks": 2, "Small Scratches": 2, "Other": 6 }, 
        "pcb1": { "BACKGROUND": 0, "Bent": 1, "Melt": 3, "Scratch": 2, "Missing": 5 }, 
        "pcb2": { "BACKGROUND": 0, "Bent": 1, "Melt": 3, "Scratch": 2, "Missing": 5 }, 
        "pcb3": { "BACKGROUND": 0, "Bent": 1, "Melt": 3, "Scratch": 2, "Missing": 5 }, 
        "pcb4": { "BACKGROUND": 0, "Burnt": 3, "Scratch": 2, "Missing": 5, "Damage": 1, "Extra": 4, "Wrong Place": 5, "Dirt": 4 }, 
        "pipe_fryum": { "BACKGROUND": 0, "Burnt": 3, "Corner And Edge Breakage": 1, "Different Colour Spot": 2, 
                       "Middle Breakage": 1, "Similar Colour Spot": 2, "Small Scratches": 2, "Stuck Together": 4, "Small Cracks": 2, "Other": 6 }
    },
    "mvtec": {
        'bottle': {'good': 0, 'contamination': 3, 'broken_small': 1, 'broken_large': 1},
        'cable': {'poke_insulation': 2, 'cut_outer_insulation': 1, 'missing_wire': 4, 'good': 0, 
                  'missing_cable': 4, 'bent_wire': 5, 'cable_swap': 4, 'cut_inner_insulation': 1, 'combined': 6},
        'capsule': {'scratch': 2, 'poke': 1, 'faulty_imprint': 2, 'good': 0, 'crack': 1, 'squeeze': 1},
        'carpet': {'good': 0, 'hole': 1, 'cut': 1, 'thread': 3, 'metal_contamination': 3, 'color': 2},
        'grid': {'good': 0, 'broken': 1, 'thread': 3, 'metal_contamination': 3, 'glue': 3, 'bent': 5},
        'hazelnut': {'good': 0, 'hole': 1, 'cut': 1, 'print': 2, 'crack': 1},
        'leather': {'poke': 1, 'good': 0, 'fold': 2, 'cut': 1, 'color': 2, 'glue': 3},
        'metal_nut': {'scratch': 2, 'good': 0, 'flip': 4, 'color': 2, 'bent': 5},
        'pill': {'scratch': 2, 'faulty_imprint': 2, 'pill_type': 4, 'good': 0, 'contamination': 3, 'combined': 6, 'color': 2, 'crack': 1},
        'screw': {'scratch_neck': 2, 'good': 0, 'scratch_head': 2, 'manipulated_front': 4, 'thread_side': 5, 'thread_top': 5},
        'tile': {'gray_stroke': 2, 'oil': 3, 'rough': 2, 'good': 0, 'glue_strip': 3, 'crack': 1},
        'toothbrush': {'good': 0, 'defective': 5},
        'transistor': {'good': 0, 'misplaced': 4, 'cut_lead': 1, 'damaged_case': 1, 'bent_lead': 5},
        'wood': {'scratch': 2, 'liquid': 3, 'good': 0, 'hole': 1, 'combined': 6, 'color': 2},
        'zipper': {'broken_teeth': 1, 'rough': 2, 'good': 0, 'fabric_interior': 3, 'split_teeth': 1, 'squeezed_teeth': 1, 'fabric_border': 3, 'combined': 6}
    },
    'mpdd':{
        'bracket_black':{'hole':1,'good':0,'scratches':2},
        'bracket_brown':{'bend_and_parts_mismatch':4,'parts_mismatch':4,'good':0},
        'bracket_white':{'defective_painting':2,'good':0,'scratches':2},
        'connector':{'parts_mismatch':4,'good':0},
        'metal_plate':{'major_rust':3,'total_rust':3,'good':0,'scratches':2},
        'tubes':{'anomalous':1,'good':0}
    },
    'btad':{
        '01':{'ko':1,'ok':0},
        '02':{'ko':1,'ok':0},
        '03':{'ko':1,'ok':0}
    },
    'SDD':{
        'electrical commutators':{'good':0,'defect':1}
    },
    'DTD':{
        'Woven_001':{'good':0,'bad':1},
        'Woven_127':{'good':0,'bad':1},
        'Woven_104':{'good':0,'bad':1},
        'Stratified_154':{'good':0,'bad':1},
        'Blotchy_099':{'good':0,'bad':1},
        'Woven_068':{'good':0,'bad':1},
        'Woven_125':{'good':0,'bad':1},
        'Marbled_078':{'good':0,'bad':1},
        'Perforated_037':{'good':0,'bad':1},
        'Mesh_114':{'good':0,'bad':1},
        'Fibrous_183':{'good':0,'bad':1},
        'Matted_069':{'good':0,'bad':1}
    },
    'DAGM_KaggleUpload':{
        'Class1':{'good':0,'scratch':1},
        'Class2':{'good':0,'metal_knot':2},
        'Class3':{'good':0,'scratch':1},
        'Class4':{'good':0,'metal_knot':2},
        'Class5':{'good':0,'hole':3},
        'Class6':{'good':0,'weaving_defect':4},
        'Class7':{'good':0,'metal_knot':2},
        'Class8':{'good':0,'scratch':1},
        'Class9':{'good':0,'metal_knot':2},
        'Class10':{'good':0,'scratch':1}
    }
}

global_defect_classes = {  
    "visa":
    {
            # "BACKGROUND": [0, "An image of the object in its normal condition without any defects."],  
            # "Chunk Of Material Missing": [1, "An image showing a chunk of material missing from the object."],  
            # "Damaged Corner of Packaging": [2, "An image displaying a damaged corner of packaging."],  
            # "Weird Candle Wick": [3, "An image of a candle with a weird or unusual wick."],  
            # "Different Colour Spot": [4, "An image showing a spot of different color on the object."],  
            # "Extra Material": [5, "An image displaying extra material on the object."],  
            # "Foreign Particles": [6, "An image showing foreign particles on the object."],  
            # "Misshape": [7, "An image of an object that is misshaped."],  
            # "Bubble": [8, "An image showing a bubble on the surface of the object."],  
            # "Discolor": [9, "An image displaying discoloration on the object."],  
            # "Scratch": [10, "An image showing scratches on the object."],  
            # "Leak": [11, "An image displaying a leak from the object."],  
            # "Burnt": [12, "An image of an object that is burnt."],  
            # "Corner or Edge Breakage": [13, "An image showing breakage at the corner or edge of the object."],  
            # "Stuck Together": [14, "An image displaying objects that are stuck together."],  
            # "Middle Breakage": [15, "An image showing breakage in the middle of the object."],  
            # "Small Holes": [16, "An image showing small holes in the object."],  
            # "Small Cracks": [17, "An image displaying small cracks on the object."],  
            # "Bent": [18, "An image of an object that is bent."],  
            # "Melt": [19, "An image showing a melted part of the object."],  
            # "Missing": [20, "An image of an object with a part missing."],  
            # "Damage": [21, "An image displaying damage on the object."],  
            # "Wrong Place": [22, "An image showing something in the wrong place on the object."],  
            # "Dirt": [23, "An image displaying dirt on the object."],  
            # "Other": [24, "An image showing an unspecified or unusual defect on the object."],
            # "Similar Colour Spot": [25, "An image showing a spot of similar color on the object."]
            "Normal Condition": [0, "An image showing the object in its standard, defect-free state with no abnormalities visible."],
            "Structural Breakage": [1, "An image revealing physical fractures or missing components including edge/corner breaks, middle fractures, or chunk losses that compromise structural integrity."],
            "Surface Imperfection": [2, "An image displaying surface-level anomalies such as scratches, discoloration, or irregular spots contrasting with the object's normal appearance."],
            "Thermal Damage": [3, "An image showing evidence of heat-related defects including burnt marks, melted components, or wax leakage caused by thermal exposure."],
            "Material Anomaly": [4, "An image containing unexpected material characteristics like stuck components, foreign particles, extra wax deposits, or bubble formations."],
            "Assembly Defect": [5, "An image demonstrating improper manufacturing outcomes including missing parts, misplaced components, or abnormal assembly configurations."],
            "Compound Defects": [6, "An image presenting multiple concurrent defect types requiring comprehensive quality assessment."]
    },
    "mvtec":{ 
        "Normal": [0, "An image of the object in its normal condition without any defects."], 
        "Structural Damage": [1, "An image showing structural damage including cracks, breaks, or deformation altering the object's original shape."], 
        "Surface Defect": [2, "An image revealing surface anomalies such as scratches, dents, or discoloration contrasting with intact areas."], 
        "Material Contamination": [3, "An image displaying foreign material contamination like stains, particles, or liquid residues on the surface."], 
        "Assembly Error": [4, "An image depicting missing components, misplaced parts, or orientation errors in object assembly."], 
        "Functional Failure": [5, "An image demonstrating functional failures including bent connectors or broken mechanisms."], 
        "Combined Defects": [6, "An image containing multiple coexisting defect types requiring comprehensive inspection."] 
    },
    "mpdd":{'Normal':[0,"An image of the object in its normal condition without any defects."],
            'Structural Damage':[1,"An image showing structural damage including holes, breaks or deformation altering the object's original shape."],
            'Surface Defect':[2,"An image revealing surface anomalies such as scratches, dents or defective painting contrasting with intact areas."],
            'Material Contamination':[3,"An image displaying rust or chemical corrosion affecting the material integrity."],
            'Assembly Error':[4,"An image depicting assembly errors including parts mismatch or incorrect component orientation."]
    },
    "btad":{'Normal':[0,"An image of the object in its normal condition without any defects."],
            'Defect':[1,"An image showing an abnormal condition with defects or functional failures."]
    },
    "SDD":{'Normal':[0,"An image of an electrical commutator in its normal operational condition without any faults."],
           'Defect':[1,"An image of an electrical commutator exhibiting defects such as physical damage, contamination, or functional failure."]
    },
    "DTD":{'Normal':[0,"an image of defect-free material in normal condition."],
           'Material_Defect':[1,"an image showing material flaws such as weaving issues, stratification problems, blotches, or structural damage."]
    },
    "DAGM_KaggleUpload":{'Normal': [0, "an image of a defect-free product."],
     'scratch': [1, "an image of a product with linear surface damage."],
     'metal_knot': [2, "an image of a product with deformed metallic protrusions."],
     'hole': [3, "an image of a product with structural perforations."],
     'weaving_defect': [4, "an image of a product with irregular textile patterns."]
     },
}  

class2name_map = {
    "visa":{
        "candle":"candle",
        "capsules":"capsules",
        "cashew":"cashew",
        "chewinggum":"chewinggum",
        "fryum":"fryum",
        "macaroni1":"macaroni",
        "macaroni2":"macaroni",
        "pcb1":"pcb",
        "pcb2":"pcb",
        "pcb3":"pcb",
        "pcb4":"pcb",
        "pipe_fryum":"fryum"
    }
}
def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id



class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        # c_max = 0
        # for cls_name,defect_types in id2class_map[dataset_name].items():
        #     c_max = len(defect_types) if len(defect_types) > c_max else c_max
        # self.c_max = c_max
        # self.id2class_map = id2class_map[dataset_name]
        self.dataset_name = dataset_name
    def __len__(self):
        return self.length
    def convert_mask(self,img_mask,cls_name,specie_name):
        if self.dataset_name == 'visa':
            vals = [val for val in local2global_id_map[self.dataset_name][cls_name].values()]
            convert_mask = np.array(vals).astype(img_mask.dtype)
            img_mask = convert_mask[img_mask]
        elif self.dataset_name == 'mvtec':
            global_id = local2global_id_map[self.dataset_name][cls_name][specie_name]
            img_mask[img_mask == 255] = global_id
        return img_mask
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        # object_category = data['object_category']
        # num_cls = len(id2class_map[self.dataset_name][cls_name].items())-1 # exclude background
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                # img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L'))>0  #[h,w]
                # img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert("RGB"))  #img_mask = cv2.imread(os.path.join(self.root, mask_path))[:, :, 0] 
                img_mask = img_mask[:, :, 0]
                img_mask = self.convert_mask(img_mask,cls_name,specie_name)
                # img_mask-=1
                # img_mask[img_mask==-1] = 255
                img_mask = Image.fromarray(img_mask) 
                # if os.path.basename(img_path)=='069.JPG':
                #     print(f"convert_mask:{convert_mask},gt_cls:{np.unique(img_mask)}")
        # transforms
        img = self.transform(img) if self.transform is not None else img

        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask

        img_mask = torch.as_tensor(np.array(img_mask)).long() if img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name],}
                # 'object_category':object_category}