FROM agrigorev/zoomcamp-model:2025

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile

WORKDIR /code
COPY pyproject.toml uv.lock ./
RUN /root/.local/bin/uv sync --frozen --no-dev

COPY predict_v2.py ./
EXPOSE 9696

CMD ["/root/.local/bin/uv", "run", "--no-sync", "uvicorn", "predict_v2:app", "--host=0.0.0.0", "--port=9696"]


