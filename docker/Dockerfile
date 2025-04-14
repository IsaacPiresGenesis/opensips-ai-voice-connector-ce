FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y procps iproute2 net-tools \
    procps iproute2 net-tools socat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV EVENT_PORT=50060

EXPOSE 50060

CMD ["python3", "-u", "src/main.py"]
