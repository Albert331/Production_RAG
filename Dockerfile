FROM python:3.11-slim

WORKDIR /production_api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY production_api/ .

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]