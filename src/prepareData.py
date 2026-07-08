from torch.utils.data import Dataset
from torchvision import transforms
import random
import os
from PIL import Image

class PrepareDataSet(Dataset):
    
    def __init__(self, root_dir, split="train", scale_factor=4, patch_size=128, seed=42):
        '''
        Lee las imagenes png del dataset en el root_dir, divide el conjunto en un 80%
        de entrenaimento y el 20% de validación.

        Para cada imagen se tiene su versión en HR (High resolution) y LR(Low resolution)
        El escalado esta diseñado para multiplicar por 4 la resolucion original.
        El entrenamiento será así, si se quiere un escalado a 8, cambiar el parametro pero
        requiere mas gpu y tiempo.

        La semilla es simplemente para poder recrear un entrenamiento.
        El patch_size es el tamaño en pixeles de cada parche HR (en este caso 64x64)
        '''
        self.root_dir = root_dir
        self.scale = scale_factor
        self.patch_size = patch_size
        self.lr_patch = patch_size // scale_factor
        
        all_files = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
        random.seed(seed)
        random.shuffle(all_files) #Para aleatoriedad en el entrenamiento y validacion
        split_index = int(0.8 * len(all_files))

        if split == "train":
            self.file_list = all_files[:split_index]
        else:
            self.file_list = all_files[split_index:]

        #Toma un recorte aleatorio del mataño patch_size x patch_size (64x64) de la imagen HR completa
        self.hr_transform = transforms.Compose([transforms.RandomCrop(patch_size),transforms.ToTensor()])

        #Redimensiona bicubicamente el parche HR recortaso a un parche LR de lr_patch x lr_patch (16 x 16)
        self.lr_transform = transforms.Compose([transforms.Resize(self.lr_patch, interpolation=Image.BICUBIC), transforms.ToTensor()])
    
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, idx): #No se usa explicitamente pero es ocupado para iterar sobre el dataset
        '''
        Encuentra la imagen en el dataset dado idx, toma un corte aleatorio de patch_size x patch_size, se guarda en hr,
        y del anterior hr se genera su version en lr, guardandolo en lr.

        Retorna:
            Un diccionario con dos claves:

            "lr": tensor de baja resolución ([3, lr_patch, lr_patch]).

            "hr": tensor de alta resolución ([3, patch_size, patch_size]).
        '''
        img_name = self.file_list[idx]
        hr_path = os.path.join(self.root_dir, img_name)
        hr_image = Image.open(hr_path).convert("RGB") #necesitamos rgn
        
        hr_patch = self.hr_transform(hr_image)
        lr_patch = self.lr_transform(transforms.ToPILImage()(hr_patch))
        
        return {"lr": lr_patch, "hr": hr_patch}