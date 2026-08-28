# Imagem da API de previsão de chuva.
#
# Build em dois estágios para que as bibliotecas de build não fiquem na
# imagem final. O resultado fica em torno de 700 MB — grande para uma
# API, pequeno para qualquer coisa com PyTorch dentro.
#
#   docker build -t rain-api .
#   docker run -p 8000:8000 rain-api

# ---------------------------------------------------------------- build
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# O torch vem do índice CPU-only. A versão do PyPI arrasta as
# bibliotecas CUDA (mais de 2 GB) e não cabe em free tier nenhum.
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------------------- runtime
FROM python:3.11-slim

# Usuário sem privilégios: se a aplicação for comprometida, o atacante
# não é root dentro do contêiner.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.pt \
    DB_PATH=/app/data/monitoring.db

COPY app/ ./app/
COPY models/ ./models/

# O SQLite é criado em runtime; o diretório precisa existir e pertencer
# ao appuser.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# O healthcheck usa o próprio /health da API. Sem curl na imagem, então
# vai por urllib mesmo.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Um único worker, de propósito: o modelo vive na memória do processo e
# é alterado in-place pelo /update. Com vários workers, cada um teria a
# sua cópia e as atualizações divergiriam entre eles.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
