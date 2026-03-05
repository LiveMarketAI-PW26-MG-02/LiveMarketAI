FROM r-base:4.3.2

LABEL system="Liquidity Blackout Storm Defense"
LABEL component="r-node"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libzmq3-dev libprotobuf-dev protobuf-compiler cmake \
    && rm -rf /var/lib/apt/lists/*

RUN Rscript -e "install.packages(c('arrow','data.table','MASS','robust','mvtnorm','fitdistrplus'), repos='https://cloud.r-project.org')"

WORKDIR /app
COPY . .

CMD ["Rscript", "r_statistical_node.R"]
