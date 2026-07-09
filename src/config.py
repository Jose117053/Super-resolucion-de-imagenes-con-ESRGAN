import torch

# RUTAS Y DIRECTORIOS
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "Flickr2K")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "images", "ciudad.jpg")
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, "scaledImages", "ciudadX4_5epocas_lousy.jpg")

# HARDWARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 # Número de núcleos del CPU dedicados a cargar datos, depende puramente de la capacidad de hardware del ordenador que se use
                # El numero de workers afecta a la cpu, tengo 8 nuecleos, asi que pongo 4 workers    

# HIPERPARÁMETROS DE ENTRENAMIENTO
NUM_EPOCHS = 150
BATCH_SIZE = 10        # Depende de la VRAM de la GPU, con 4gb de VRAM 10 es el maximo
LR_INITIAL = 1e-4      # Learning rate inicial sugerido por el paper (En principio yo tenia 5e-5)
BETAS = (0.9, 0.999)   # Parámetros para el optimizador Adam

# PARÁMETROS DEL DATASET (PREPARACIÓN)
SCALE_FACTOR = 4       # Factor de escalado, el modelo actual solo soporta 4 como input (Todo depende del hardware)
PATCH_SIZE = 128       # Tamaño de los recortes de imagen para entrenar

# ARQUITECTURA DEL GENERADOR (RRDBNet)
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_FEATURES = 64
GROWTH_CHANNELS = 32
NUM_BLOCKS = 23

# CONTROL DE ENTRENAMIENTO CONTINUO
CONTINUAR_ENTRENAMIENTO = False
CHECKPOINT_PATH = f"{CHECKPOINTS_DIR}/checkpoint_epoch_40.pth"

# PARÁMETROS DE INFERENCIA
TILE_SIZE = 256        # Tamaño de mosaico para procesar imágenes grandes sin saturar VRAM
OVERLAP = 32           # Solapamiento entre mosaicos para evitar bordes cortados
