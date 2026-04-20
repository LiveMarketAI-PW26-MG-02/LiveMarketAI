FROM python:3.11-slim
LABEL system="ModelDrift Sabotage Defense"
LABEL component="federated-aggregator"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../proto/service.proto /proto/service.proto
RUN python -m grpc_tools.protoc \
        -I /proto \
        --python_out=. \
        --grpc_python_out=. \
        /proto/service.proto

COPY . .
EXPOSE 50051

CMD ["python", "aggregator_server.py"]
