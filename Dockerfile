FROM python:3.10

WORKDIR /app

COPY app /app

RUN pip install -r service/requirements.txt

EXPOSE 5000

CMD ["python", "service/app.py"]
