# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the Docker container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script on container startup
CMD ["python", "./training_model.py"]
