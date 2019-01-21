FROM python:3.6

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY tf_model /app
COPY model_server.py /app

CMD [ "python", "model_server.py" ]