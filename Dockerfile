#Construir: docker-compose build
#Ejecutar: docker-compose up
#Acceder: http://localhost:8888

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    jupyter

COPY src/ ./src/
COPY gan.ipynb ./gan.ipynb
COPY upscale.py ./upscale.py
COPY resources/ ./resources/

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
    "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''",]

