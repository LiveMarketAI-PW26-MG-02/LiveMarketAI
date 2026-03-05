FROM eclipse-temurin:21-jdk-jammy

LABEL system="Wash Trading Illuminator"
LABEL component="java-node"

RUN apt-get update && apt-get install -y --no-install-recommends maven && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Proto compile (assumes gRPC Maven plugin)
RUN mvn package -DskipTests -q

CMD ["java", "-jar", "target/fgce-java-node-1.0.0-SNAPSHOT.jar"]
