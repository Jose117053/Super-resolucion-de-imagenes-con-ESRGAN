import torch

#En un el ESRGAN tradicional no se considera la perdida de color.
#Pero yo percibia que se perdia tras pasar una imagen sobre el modelo ya entrenado.
#Así que decidí extender el entrenamiento para que considere la perdida de color

#Es necesaria la conversion para extraer la informacion de luminencia(Y) y  crominancia (Cb y Cr)
#Al separarlos podemos penalizar errores de color sin afectar la parte de brillo
def rgb_to_ycbcr_batch(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convierte un tensor de forma (B, 3, H, W) en espacio RGB (0..1) 
    a YCbCr (también normalizado en 0..1).


    Utilizamos la conversión estándar:
      Y  =  0.299 R + 0.587 G + 0.114 B
      Cb = -0.168736 R - 0.331264 G + 0.5    B + 0.5
      Cr =  0.5    R - 0.418688 G - 0.081312 B + 0.5
    """
    # forma (B,3,H,W) B imagenes con los 3 colores con una altura H y anchura W
    R = rgb_tensor[:, 0:1, :, :]  #Nada mas nos quedamos con el color rojo
    G = rgb_tensor[:, 1:2, :, :]  #Nada mas nos quedamos con el color verde
    B = rgb_tensor[:, 2:3, :, :]  #Nada mas nos quedamos con el color azul

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5

    return torch.cat([Y, Cb, Cr], dim=1)  # (B,3,H,W)