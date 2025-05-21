FROM ubuntu:22.04

# Diretório de trabalho do container
WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip

# Instalar ou atualizar o NumPy para garantir que seja >= 1.24
RUN pip install --no-cache-dir numpy>=1.24

# Aggiorna il sistema e installa le dipendenze necessarie
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    python3.9 \
    python3-distutils \
    python3-pip \
    ffmpeg

RUN pip install --no-cache-dir --upgrade setuptools pip

# Installa openai-whisper
RUN pip install -U openai-whisper

# Copie o arquivo requirements.txt para o container
COPY requirements.txt /app/

# Instale as dependências do projeto
RUN pip install -r requirements.txt

# Copie o restante do código da aplicação para o container
COPY . /app/

# Exponha a porta que o Flask vai rodar (normalmente 5000)
EXPOSE 5000

RUN python3 --version

# # Defina o comando para rodar a aplicação
CMD ["python3", "transcription.py"]