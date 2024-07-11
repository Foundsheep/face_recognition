import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from configs import Config

def load_data(root_folder, shuffle, batch_size, num_workers, read_step, is_train):
    ds = CroppedFaceDataset(root_folder=root_folder,
                            read_step=read_step,
                            is_train=is_train,)
    try:  
        dl = DataLoader(ds,
                        shuffle=shuffle,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        persistent_workers=True)
    except:
        dl = None
    return dl

class CroppedFaceDataset(Dataset):
    def __init__(self, root_folder, read_step=5, is_train=True) -> None:
        super().__init__()
        self.root_folder = root_folder
        self.read_step = read_step
        self.is_train = is_train
        
        returned_stuff = self._read_paths()
        self.image_list = returned_stuff[0]
        self.label_list = returned_stuff[1]
        self.label_to_id_dict = returned_stuff[2]
        self.id_to_label_dict = returned_stuff[3]
        self.label_to_name_dict = returned_stuff[4]
        self.name_to_label_dict = returned_stuff[5]
        self.transforms = self._get_transforms()

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), f"[{len(self.image_list) = }], [{len(self.label_list) = }]"
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image, label = None, None
        image_path = self.image_list[idx]

        image = Image.open(image_path)
        image = np.array(image)
        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = self.label_list[idx]
        label = torch.Tensor([label]).type_as(image)

        return image, label
    
    def _read_paths(self):
        
        # returning stuff
        image_list = []
        label_list = []
        label_to_id_dict = {}
        id_to_label_dict = {}
        label_to_name_dict = {}
        name_to_label_dict = {}

        root_p = Path(self.root_folder)
        folders = root_p.glob("*/")
        temp_label = 0

        for folder in folders:
            folder_name = folder.parts[-1]
            person_name, person_id = folder_name.split("_")[:2]
            images = folder.glob("*.png")
            images = list(images)
            images = images[::self.read_step]
            
            # 사번이 처음 인식되면
            if person_id not in id_to_label_dict.keys():
                id_to_label_dict[person_id] = temp_label
                label_to_id_dict[temp_label] = person_id

                # 사번이 처음인데, 같은 이름이 있을 경우
                if person_name in name_to_label_dict.keys():
                    # 새롭게 넘버링된 person_name    
                    name_number = 2
                    person_name = person_name + str(name_number).zfill(2)
                    while person_name in name_to_label_dict.keys():
                        name_number += 1
                        person_name = person_name + str(name_number).zfill(2)

                name_to_label_dict[person_name] = temp_label
                label_to_name_dict[temp_label] = person_name
                
                # 다음 라벨 준비
                temp_label += 1
            else:
                print(f"[{person_name} / {person_id}] has been recognised before")

            for image in images:
                image_list.append(str(image))
                label_list.append(id_to_label_dict[person_id])
            
        # 저장
        prefix = Path(__file__).absolute().parent.parent / "training_jsons"
        # prefix = "./training_jsons/"

        if not prefix.exists():
            prefix.mkdir(parents=True)
            print(f"prefix : [{prefix}] made!!!")

        label_to_id_dict_path = prefix / "label_to_id.json"
        id_to_label_dict_path = prefix / "id_to_label.json"
        label_to_name_dict_path = prefix / "label_to_name.json"
        name_to_label_dict_path = prefix / "name_to_label.json"
        json_paths = [label_to_id_dict_path, id_to_label_dict_path, label_to_name_dict_path, name_to_label_dict_path]
        dict_list = [label_to_id_dict, id_to_label_dict, label_to_name_dict, name_to_label_dict]

        for json_path, dict_obj in zip(json_paths, dict_list):
            with open(str(json_path), "w") as j:
                print("====", json_path)
                json.dump(dict_obj, j, ensure_ascii=False, indent=4)
                print(f"[{json_path}] saved!")

        return image_list, label_list, label_to_id_dict, id_to_label_dict, label_to_name_dict, name_to_label_dict
    
    def _get_transforms(self):
        if self.is_train:
            transforms = A.Compose([
                A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                A.HorizontalFlip(),
                A.Blur(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ])
        else:
            transforms = A.Compose([
                A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ])
        return transforms 
    

if __name__ == "__main__":
    file_folder = Path(__file__).absolute().parent
    os.chdir(file_folder)
    ds = CroppedFaceDataset(root_folder="./image_saved")
    print(f"{len(ds) = }")
    for d in ds:
        break

    print(f"{d[0].size()}")
    print(f"{d[1].size()}")