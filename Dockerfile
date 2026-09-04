FROM python:3.12-slim

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    autoconf \
    automake \
    libtool \
    make \
    gcc \
    g++ \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# GET OFFICIAL RNNOISE SOURCE
# ============================================================

RUN git clone \
    --depth 1 \
    https://github.com/xiph/rnnoise.git \
    /app/rnnoise


# ============================================================
# BUILD RNNOISE
# ============================================================

WORKDIR /app/rnnoise

RUN chmod +x autogen.sh download_model.sh

RUN ./autogen.sh && \
    ./configure && \
    make


# ============================================================
# STREAMLIT APP
# ============================================================

WORKDIR /app

COPY app.py .
COPY audio ./audio


# ============================================================
# RUN
# ============================================================

EXPOSE 8501

CMD streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8501}