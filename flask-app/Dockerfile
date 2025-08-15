# Use Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt
# Copy the entire content of the flask-app folder to the container
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "app.py"]
