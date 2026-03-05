FROM python:3.11-slim

LABEL system="StreamHijack Specter Defense"
LABEL component="python-node"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate gRPC stubs from proto
COPY ../proto/service.proto /proto/service.proto
RUN python -m grpc_tools.protoc \
        -I /proto \
        --python_out=. \
        --grpc_python_out=. \
        /proto/service.proto

EXPOSE 50051 8815 5570

CMD ["python", "federated_coordinator.py"]
