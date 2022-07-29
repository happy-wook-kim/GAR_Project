import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class GADataset(Dataset):    
    def __init__(self, img_size, dataset_path = './dataset/GA/'):
        if not img_size % 2 == 0:
            print("image size 는 짝수로 맞춰주세요.")
            raise
        self.dataset_path = dataset_path
        self.images = self.get_images(self.dataset_path)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self,idx):
        image_path = self.images[idx]

        image = Image.open(image_path)
        image = self.transform(image)
        gender_name = image_path.split('/')[-3]
        age_name = image_path.split('/')[-2]
        age = 0
        gender = 0

        if age_name == "student" :
            age = 0
        elif age_name == "young" :
            age = 1
        elif age_name == "middle" :
            age = 2
        elif age_name == "old" :
            age = 3
        else :
            print('age name : ',age_name)
            raise ValueError()

        if gender_name == "male":
            gender = 0
        elif gender_name == "female" :
            gender = 1
        else :
            print('gender name ', gender_name)
            raise ValueError()

        return image,gender,age
 
    def get_images(self, dataset_path):
        images = []
        images += self.get_directory_files(os.path.join(dataset_path,'male/student'))
        images += self.get_directory_files(os.path.join(dataset_path,'male/young'))
        images += self.get_directory_files(os.path.join(dataset_path,'male/middle'))
        images += self.get_directory_files(os.path.join(dataset_path,'male/old'))

        images += self.get_directory_files(os.path.join(dataset_path,'female/student'))
        images += self.get_directory_files(os.path.join(dataset_path,'female/young'))
        images += self.get_directory_files(os.path.join(dataset_path,'female/middle'))
        images += self.get_directory_files(os.path.join(dataset_path,'female/old'))

        return images

    def get_directory_files(self, dir_path):
       file_list = os.listdir(dir_path)
       return [ os.path.join(dir_path,file_name) for file_name in file_list if not 'DS' in file_name]


class FrontDataset(Dataset):    
    def __init__(self, img_size, dataset_path = './dataset/Front/'):
        if not img_size % 2 == 0:
            print("image size 는 짝수로 맞춰주세요.")
            raise
        self.dataset_path = dataset_path
        self.images = self.get_images(self.dataset_path)
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self,idx):
        image_path = self.images[idx]

        image = Image.open(image_path)
        image = self.transform(image)
        label = 0
        image_name = image_path.split('/')[-2] 
        if 'front' in image_name:
            label = 0
        elif 'side' in image_name :
            label = 1
        elif 'back' in image_name :
            label = 2
        elif 'hat' in image_name :
            label = 3
        else :
            label = 4
        if label > 0 :
            label = 1
        return image,label
 
    def get_images(self, dataset_path):
        images = []
        images += self.get_directory_files(os.path.join(dataset_path,'front'))
        images += self.get_directory_files(os.path.join(dataset_path,'side'))
        images += self.get_directory_files(os.path.join(dataset_path,'back'))
        images += self.get_directory_files(os.path.join(dataset_path,'unknown'))
        images += self.get_directory_files(os.path.join(dataset_path,'hat'))
        return images

    def get_directory_files(self, dir_path):
       file_list = os.listdir(dir_path)
       return [ os.path.join(dir_path,file_name) for file_name in file_list if not 'DS' in file_name]

if __name__ == "__main__":
    print('Dataset.py')
    parser = argparse.ArgumentParser(description='Datasets')

    parser.add_argument("--model_type", default="GA", type=str, help="Specify model type")
    parser.add_argument("--image_size", default=64, type=int, help='Specify image size.')
    args = parser.parse_args()
    img_size = args.image_size
    model_type = args.model_type

    if not model_type == "Front" and not model_type == "GA":
        print("올바른 모델 타입을 입력하세요.")
        raise

    if not img_size % 2 == 0:
        print("image size 는 짝수로 맞춰주세요.")
        raise

    print(model_type + "Dataset\nimage size: ",img_size)