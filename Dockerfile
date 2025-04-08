
# Use an official Python runtime as a parent image
FROM python:3.11

WORKDIR /app

# Copy the entire project into the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python", "train_model.py"]
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:10000"]
