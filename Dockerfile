# Use base image for Python 3.10
FROM python:3.10-slim-bookworm

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR .

# Copy the requirements file first for better caching
COPY requirements.txt .

# Install required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Pre-download the model to avoid runtime issues
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; model_name='sentence-transformers/all-MiniLM-L6-v2'; AutoModel.from_pretrained(model_name); AutoTokenizer.from_pretrained(model_name)"
RUN python3 -c "from transformers import BartTokenizer, BartForConditionalGeneration; model_name='facebook/bart-large-cnn'; BartForConditionalGeneration.from_pretrained(model_name); BartTokenizer.from_pretrained(model_name)"

# Copy application code into the app directory
COPY . .

# Make the start script executable
RUN chmod +x scripts/start.sh

# Define the container's entry point
ENTRYPOINT ["scripts/start.sh"]

EXPOSE 8501
