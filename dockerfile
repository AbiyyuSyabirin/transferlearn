# Gunakan image Python 3.9 sebagai base image
FROM python:3.10

# Set working directory di dalam container
WORKDIR /app

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi ke dalam container
COPY . .

# Ekspos port 8000
EXPOSE 8000

# Jalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]