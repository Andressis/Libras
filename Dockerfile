# Imagem base do Python 3.10 slim (leve, mas requer libs extras)
FROM python:3.10-slim

# Evita prompts interativos durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório da aplicação
WORKDIR /app

# Instalar dependências do sistema (OpenCV, numpy e compilação de pacotes)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  # limpeza para reduzir o tamanho da imagem

# Copiar e instalar dependências do Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código da aplicação
COPY . .

# Comando para rodar com Gunicorn, usando a variável de ambiente $PORT (Heroku)
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000}"]
