FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY python ./python
COPY tools ./tools

RUN python -m pip install --no-cache-dir -e .

CMD ["python", "-m", "pytest"]
