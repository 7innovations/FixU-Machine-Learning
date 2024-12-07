# Gunakan Python 3.11 sebagai base image
FROM python:3.11-slim


# Atur direktori kerja di dalam container
WORKDIR /app

# Install dependencies sistem yang diperlukan untuk TensorFlow dan Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt ke dalam container
COPY requirements.txt requirements.txt
COPY .env .env

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Tambahkan debug print untuk memastikan model ada
RUN ls -l models/

# Set environment variable untuk Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV PORT=8080

# Expose port 8080 untuk akses API
EXPOSE 8080

# Command untuk menjalankan aplikasi
CMD exec gunicorn --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 600 \
    --log-level debug \
    app:app