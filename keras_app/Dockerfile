FROM python:3.6

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY . .
COPY tf_model/ /app

COPY model_server.py /app
CMD [ "python", "./model_server.py" ]