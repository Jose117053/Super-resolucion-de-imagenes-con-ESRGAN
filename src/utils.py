import torch
import numpy as np
from PIL import Image

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




def super_resolve_image(model,img_path,device,scale=4,tile_size=256,overlap=32):
    """
    Aplica la super resolucion a partir de un modelo ya entrenado

    Parametros:
        model: El modelo RRDBNet Entrenado.
        img_path: Direccion a la imagen que se le aplicará el escalado.
        device: Se ocupara para realizar el trabajo (cpu o gpu).
        scale: Factor de escalado, el modelo actual solo soporta 4 como input.
        tile_size: Tamaño de cada mosaico que se procesa de manera independiente.
        overlap: Longitud de cada mosaico para que se extienda sobre otro, sirve para evitar bordes visibles y tener una continuidad de texturas.

    """

    #Poner el modelo en modo evaluacion
    model.eval()
    lr_image = Image.open(img_path).convert("RGB")
    lr_np = np.array(lr_image).astype(np.float32) / 255.0  # Convertir en un array 
    h_lr, w_lr, _ = lr_np.shape #alto y largo
    h_hr, w_hr = h_lr * scale, w_lr * scale  # Dimensiones de salida

    output_sum = np.zeros((h_hr, w_hr, 3), dtype=np.float32)
    count_map = np.zeros((h_hr, w_hr, 3), dtype=np.float32)

    def process_tile(x0, y0, x1, y1):
        '''
        Recorta un mosaico de la imagen LR, lo procesa con el modelo y devuelve el
        mosaico HR resultante como arreglo NumPy en [0,1].

        Parametros:
            x0, y0: Coordenadas de la esquina superior izquierda del mosaico en LR.
            x1, y1: Coordenadas de la esquina inferior derecha del mosaico en LR.

        '''
        lr_tile = lr_np[y0:y1, x0:x1, :]
        # Convierte a tensor con forma [1, 3, H, W]
        lr_tensor = torch.from_numpy(lr_tile.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Convierte el tensor de salida a NumPy en (H*scale, W*scale, 3)
        sr_np = sr_tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return sr_np

    stride = tile_size - overlap
    for y in range(0, h_lr, stride):
        for x in range(0, w_lr, stride):
            # Determina bordes del mosaico, sin exceder los límites de la image
            x_end = min(x + tile_size, w_lr)
            y_end = min(y + tile_size, h_lr)
            x0, y0 = x, y
            x1, y1 = x_end, y_end

         
            sr_tile = process_tile(x0, y0, x1, y1)
            # Calcular coordenadas en la version HR
            x0_hr, y0_hr = x0 * scale, y0 * scale
            x1_hr, y1_hr = x1 * scale, y1 * scale

            # Acumula valores de píxeles
            output_sum[y0_hr:y1_hr, x0_hr:x1_hr, :] += sr_tile
            count_map[y0_hr:y1_hr, x0_hr:x1_hr, :] += 1.0

    output_avg = output_sum / count_map
    output_img = (output_avg * 255.0).round().astype(np.uint8)     # Convierte de nuevo a uint8 en [0,255] (formato necesario para una imagen)
    return Image.fromarray(output_img)