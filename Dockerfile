FROM python:3.10

WORKDIR /usr/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

CMD ["python3", "src/main.py"]

