import io
import os
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles

from src.generator import RRDBNet
from src import config as cfg
import src.utils as utils

app = FastAPI(title="ESRGAN API", description="Image Super-Resolution API")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

generator = None

@app.on_event("startup")
async def load_model():
    global generator
    print("Cargando modelo ESRGAN en memoria...")
    
    generator = RRDBNet(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_features=cfg.NUM_FEATURES,
        growth_channels=cfg.GROWTH_CHANNELS,
        num_blocks=cfg.NUM_BLOCKS
    ).to(cfg.DEVICE)
    
    if os.path.exists(cfg.CHECKPOINT_PATH):
        print(f"Cargando pesos desde: {cfg.CHECKPOINT_PATH}")
        ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE)
        generator.load_state_dict(ckpt['G_state_dict'])
        generator.eval()
        print("Modelo cargado exitosamente.")
    else:
        print(f"ADVERTENCIA: No se encontró el checkpoint en {cfg.CHECKPOINT_PATH}")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Sirve la interfaz web (Frontend)
    """
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    """
    Recibe una imagen, la procesa con ESRGAN y la devuelve en alta resolución.
    """
    if generator is None:
        return {"error": "El modelo no se cargó correctamente."}
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_in:
        contents = await file.read()
        tmp_in.write(contents)
        tmp_in_path = tmp_in.name
        
    try:
        sr_image = utils.super_resolve_image(
            model=generator,
            img_path=tmp_in_path,
            device=cfg.DEVICE,
            scale=cfg.SCALE_FACTOR,
            tile_size=cfg.TILE_SIZE,
            overlap=cfg.OVERLAP
        )
        
        buf = io.BytesIO()
        sr_image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/jpeg")
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

@app.get("/health")
async def health_check():
    """
    Endpoint requerido por Google Cloud Run para verificar si el servidor está vivo.
    """
    return {"status": "healthy", "model_loaded": generator is not None}
