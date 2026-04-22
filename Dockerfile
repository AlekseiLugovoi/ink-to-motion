FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download rembg model so it's baked into the image (matches _SEG_MODEL in app/pipeline.py)
RUN python -c "from rembg import new_session; new_session('birefnet-general-lite')"
# RUN python -c "from rembg import new_session; new_session('birefnet-general')"  # целевая (Pro план)

COPY app/ .

CMD ["python", "app.py"]
