FROM python:3.10-slim

# Set up user to avoid running as root, which Hugging Face prefers
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the current directory contents into the container
COPY --chown=user . $HOME/app

# Install dependencies (using the CPU-only PyTorch setup we already configured)
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces routes traffic to port 7860
ENV PORT=7860
EXPOSE 7860

# Run the server
CMD ["python", "server.py"]
