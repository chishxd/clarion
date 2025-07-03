FROM python:3.13-slim

WORKDIR /app

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src/clarion /app/clarion

EXPOSE 80
CMD ["uvicorn", "clarion.app:app", "--host", "0.0.0.0", "--port", "80"]
