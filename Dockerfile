FROM python:3.10-slim

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório da aplicação
WORKDIR /app

# Instalar dependências do sistema necessárias ao OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Rodar o servidor com gunicorn
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
