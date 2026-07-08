"""
Escala imágenes usando un modelo ESRGAN previamente entrenado.

Uso:
    python upscale.py --input foto.jpg --output foto_x4.jpg
    python upscale.py --input foto.jpg --output foto_x4.jpg --checkpoint ./checkpoints/checkpoint_epoch_150.pth
    python upscale.py --input ./images/ --output ./scaledImages/   (procesa toda una carpeta)
"""

import argparse
import os
import sys
import torch

from src.generator import RRDBNet
from src.utils import super_resolve_image
import src.config as cfg


def load_model(checkpoint_path, device):
    """
    Carga el modelo generador RRDBNet y restaura los pesos desde un checkpoint.
    """
    generator = RRDBNet(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_features=cfg.NUM_FEATURES,
        growth_channels=cfg.GROWTH_CHANNELS,
        num_blocks=cfg.NUM_BLOCKS,
        scale=cfg.SCALE_FACTOR
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True) #weights_only evita carga de data maliciosa
    generator.load_state_dict(checkpoint["G_state_dict"])
    generator.eval()

    print(f"Modelo cargado desde: {checkpoint_path}")
    print(f"Device: {device}")
    return generator


def process_single_image(model, input_path, output_path, device):
    """
    Aplica super resolución a una sola imagen y la guarda en disco.
    """
    print(f"  Procesando: {input_path}")
    sr_image = super_resolve_image(
        model=model,
        img_path=input_path,
        device=device,
        scale=cfg.SCALE_FACTOR,
        tile_size=cfg.TILE_SIZE,
        overlap=cfg.OVERLAP
    )
    sr_image.save(output_path)
    print(f"  Guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ESRGAN — Escala imágenes x4 usando super resolución con redes neuronales."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta a una imagen o carpeta con imágenes para escalar."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Ruta donde guardar la imagen escalada o carpeta de salida (opcional)."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=cfg.CHECKPOINT_PATH,
        help=f"Ruta al archivo .pth del checkpoint (default: {cfg.CHECKPOINT_PATH})."
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forzar el uso de CPU aunque haya GPU disponible."
    )

    args = parser.parse_args()

    # Determinar el dispositivo
    device = torch.device("cpu") if args.cpu else cfg.DEVICE

    # Validar que el checkpoint existe
    if not os.path.isfile(args.checkpoint):
        print(f"Error: No se encontró el checkpoint en '{args.checkpoint}'.")
        print("Asegúrate de haber entrenado el modelo primero o especifica la ruta con --checkpoint.")
        sys.exit(1)

    # Cargar el modelo
    model = load_model(args.checkpoint, device)

    # Extensiones de imagen soportadas
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Caso 1: La entrada es una sola imagen
    if os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1].lower()
        if ext not in valid_extensions:
            print(f"Error: '{args.input}' no es un formato de imagen soportado.")
            sys.exit(1)

        # Si no se provee salida, usamos la carpeta de la imagen original
        if args.output is None:
            args.output = os.path.dirname(os.path.abspath(args.input)) or "."

        # Si la salida es una carpeta, generar el nombre automáticamente
        if os.path.isdir(args.output):
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{base_name}_x{cfg.SCALE_FACTOR}{ext}")
        else:
            output_path = args.output
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        process_single_image(model, args.input, output_path, device)

    # Caso 2: La entrada es una carpeta con múltiples imágenes
    elif os.path.isdir(args.input):
        # Si no se provee salida, creamos una nueva carpeta al lado de la original
        if args.output is None:
            args.output = os.path.abspath(args.input).rstrip('/') + f"_x{cfg.SCALE_FACTOR}"
            
        os.makedirs(args.output, exist_ok=True)

        images = [
            f for f in sorted(os.listdir(args.input))
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if not images:
            print(f"No se encontraron imágenes en '{args.input}'.")
            sys.exit(1)

        print(f"Se encontraron {len(images)} imágenes para escalar.\n")

        for i, img_name in enumerate(images, 1):
            input_path = os.path.join(args.input, img_name)
            base_name, ext = os.path.splitext(img_name)
            output_path = os.path.join(args.output, f"{base_name}_x{cfg.SCALE_FACTOR}{ext}")

            print(f"[{i}/{len(images)}]")
            process_single_image(model, input_path, output_path, device)

        print(f"\n¡Listo! {len(images)} imágenes escaladas x{cfg.SCALE_FACTOR} guardadas en '{args.output}'.")

    else:
        print(f"Error: '{args.input}' no es un archivo ni una carpeta válida.")
        sys.exit(1)


if __name__ == "__main__":
    main()
