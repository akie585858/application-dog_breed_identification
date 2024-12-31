import os
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torch
from PIL import Image

DOG_BREED_MEAN = [0.3915, 0.4507, 0.4752]
DOG_BREED_STD = [0.2201, 0.2218, 0.2268]

class DogDataset(Dataset):
    label_set = {'affenpinscher': 0, 'afghan_hound': 1, 'african_hunting_dog': 2, 'airedale': 3, 'american_staffordshire_terrier': 4, 'appenzeller': 5, 'australian_terrier': 6, 'basenji': 7, 'basset': 8, 'beagle': 9, 'bedlington_terrier': 10, 'bernese_mountain_dog': 11, 'black-and-tan_coonhound': 12, 'blenheim_spaniel': 13, 'bloodhound': 14, 'bluetick': 15, 'border_collie': 16, 'border_terrier': 17, 'borzoi': 18, 'boston_bull': 19, 'bouvier_des_flandres': 20, 'boxer': 21, 'brabancon_griffon': 22, 'briard': 23, 'brittany_spaniel': 24, 'bull_mastiff': 25, 'cairn': 26, 'cardigan': 27, 'chesapeake_bay_retriever': 28, 'chihuahua': 29, 'chow': 30, 'clumber': 31, 'cocker_spaniel': 32, 'collie': 33, 'curly-coated_retriever': 34, 'dandie_dinmont': 35, 'dhole': 36, 'dingo': 37, 'doberman': 38, 'english_foxhound': 39, 'english_setter': 40, 'english_springer': 41, 'entlebucher': 42, 'eskimo_dog': 43, 'flat-coated_retriever': 44, 'french_bulldog': 45, 'german_shepherd': 46, 'german_short-haired_pointer': 47, 'giant_schnauzer': 48, 'golden_retriever': 49, 'gordon_setter': 50, 'great_dane': 51, 'great_pyrenees': 52, 'greater_swiss_mountain_dog': 53, 'groenendael': 54, 'ibizan_hound': 55, 'irish_setter': 56, 'irish_terrier': 57, 'irish_water_spaniel': 58, 'irish_wolfhound': 59, 'italian_greyhound': 60, 'japanese_spaniel': 61, 'keeshond': 62, 'kelpie': 63, 'kerry_blue_terrier': 64, 'komondor': 65, 'kuvasz': 66, 'labrador_retriever': 67, 'lakeland_terrier': 68, 'leonberg': 69, 'lhasa': 70, 'malamute': 71, 'malinois': 72, 'maltese_dog': 73, 'mexican_hairless': 74, 'miniature_pinscher': 75, 'miniature_poodle': 76, 'miniature_schnauzer': 77, 'newfoundland': 78, 'norfolk_terrier': 79, 'norwegian_elkhound': 80, 'norwich_terrier': 81, 'old_english_sheepdog': 82, 'otterhound': 83, 'papillon': 84, 'pekinese': 85, 'pembroke': 86, 'pomeranian': 87, 'pug': 88, 'redbone': 89, 'rhodesian_ridgeback': 90, 'rottweiler': 91, 'saint_bernard': 92, 'saluki': 93, 'samoyed': 94, 'schipperke': 95, 'scotch_terrier': 96, 'scottish_deerhound': 97, 'sealyham_terrier': 98, 'shetland_sheepdog': 99, 'shih-tzu': 100, 'siberian_husky': 101, 'silky_terrier': 102, 'soft-coated_wheaten_terrier': 103, 'staffordshire_bullterrier': 104, 'standard_poodle': 105, 'standard_schnauzer': 106, 'sussex_spaniel': 107, 'tibetan_mastiff': 108, 'tibetan_terrier': 109, 'toy_poodle': 110, 'toy_terrier': 111, 'vizsla': 112, 'walker_hound': 113, 'weimaraner': 114, 'welsh_springer_spaniel': 115, 'west_highland_white_terrier': 116, 'whippet': 117, 'wire-haired_fox_terrier': 118, 'yorkshire_terrier': 119}
    def __init__(self, is_train=True):
        super().__init__()
        root_path = 'dataset/dog-breed-identification'
        img_path = 'dataset/dog-breed-identification/train'
        if is_train:
            file_path = os.path.join(root_path, 'train.txt')
        else:
            file_path = os.path.join(root_path, 'test.txt')

        with open(file_path, 'r') as f:
            self.files_list = f.readlines()

        self.img_list = []
        self.label_list = []

        # 获取label列表
        self.label_list = []
        for data_line in self.files_list:
            img, label = data_line.split(',')
            img = os.path.join(img_path, img+'.jpg')
            label = DogDataset.label_set[label.strip()]

            self.img_list.append(img)
            self.label_list.append(label)
        
        if is_train:
            self.transform = transforms.Compose([
                # transforms.ColorJitter(0.01, 0.01, 0.01, 0.01),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((448, 448)),
                transforms.RandomRotation(90),
                transforms.Normalize(mean=DOG_BREED_MEAN, std=DOG_BREED_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DOG_BREED_MEAN, std=DOG_BREED_STD)
            ])
            
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = self.transform(img)
        label = self.label_list[index]

        return img, label
    

if __name__ == '__main__':
    # 数据集归一化
    import tqdm
    dataset = DogDataset()
    
    mean_all = 0
    std_all = 0
    count = 0
    for img, label in tqdm.tqdm(dataset):
        mean = torch.mean(img, dim=[1, 2])
        std = torch.std(img, dim=[1, 2])

        mean_all = (count/(count+1))*mean_all + mean/(count+1)
        std_all = (count/(count+1))*std_all + std/(count+1)
        count += 1
        
    print(mean_all)
    print(std_all)
    
    # 文件路径处理与数据集分割
    # import pandas as pd
    # import numpy as np
    # data = pd.read_csv('dataset/dog-breed-identification/labels.csv')
    # data = data.values
    # cls_list = data[:, 1]
    # cls_list = np.unique(cls_list)
    # transform_label = {cls_attr:idx for idx, cls_attr in enumerate(cls_list)}
    # print(transform_label)
    
    # np.random.shuffle(data)
    # train_length = int(0.8*len(data))
    # train = data[:train_length]
    # test = data[train_length:]

    # root = 'dataset/dog-breed-identification'
    # with open(os.path.join(root, 'train.txt'), 'w') as f:
    #     for img, label in train:
    #         f.write(f'{img},{label}\n')

    # with open(os.path.join(root, 'test.txt'), 'w') as f:
    #     for img, label in test:
    #         f.write(f'{img},{label}\n')