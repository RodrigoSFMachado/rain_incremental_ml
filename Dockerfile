# Imagem da API de previsão de chuva.
#
# Build em dois estágios para manter as bibliotecas de build fora da
# imagem final. O resultado fica em torno de 700 MB: grande para uma API,
# mas pequeno para uma aplicação que inclui PyTorch.
#
#     docker build -t rain-api .
#     docker run -p 8000:8000 rain-api

# ---------------------------------------------------------------- build
FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala o PyTorch pelo índice CPU-only. A versão padrão do PyPI pode
# trazer bibliotecas CUDA, adicionando mais de 2 GB à imagem.
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------------------- runtime
FROM python:3.11-slim

# Usuário sem privilégios: se a aplicação for comprometida, o atacante
# não terá permissões de root dentro do contêiner.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.pt \
    DB_PATH=/app/data/monitoring.db

# Copia apenas os artefatos usados em runtime pela API. Os diretórios
# `training/` e `tests/` não são necessários na imagem do serviço.
COPY app/ ./app/
COPY models/ ./models/

# O SQLite é criado em runtime; o diretório precisa existir e pertencer
# ao usuário da aplicação.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# O healthcheck usa o endpoint `/health` da própria API. Como a imagem não
# inclui `curl`, a verificação é feita com `urllib`.
#
# Isso confirma apenas a liveness: o processo HTTP está respondendo. Não
# confirma readiness, pois `/health` retorna 200 mesmo sem o modelo,
# informando `"status": "degraded"` e `"model_loaded": false` no corpo.
#
# A porta fica fixa em 8000 porque é a usada por `EXPOSE` e pelo Docker
# Compose. Em plataformas que fornecem `$PORT`, como o Render, o
# healthcheck deste Dockerfile não é usado; vale o `healthCheckPath`
# definido em `render.yaml`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Usa um único worker de propósito: o modelo permanece na memória do
# processo e é alterado in-place por `/update`. Com vários workers, cada
# processo teria sua própria cópia e as atualizações poderiam divergir.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
