import os
import json

from llm_prompts_generator import llm_prompts_generate
class MVTecSolver(object):
    CLSNAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            object_category = cls_name,
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
        id2class_map = {}
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            id2class_map[cls_name]=[]
            species = os.listdir(f'{cls_dir}/test')
            for i,specie in enumerate(species):
                # if specie=="good":
                #     specie = "BACKGROUD"
                id2class_map[cls_name].append(specie)
        # with open(os.path.join(self.root,'id2class_map.json'), 'w') as f:
        #     f.write(json.dumps(id2class_map, indent=4) + "\n")
        print(id2class_map)
        return id2class_map
       
                

if __name__ == '__main__':
    runner = MVTecSolver(root='/mnt/ssd/home/jcheng/datasets/mvtec')
    # # runner.run()

    runner.get_all_defect_types()