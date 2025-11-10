# FastAPI – Adult Income Prediction Service (Docker Deployment)

This guide explains how to build and run the **Adult (Census Income)** prediction model inside Docker, using a FastAPI service.

---

## 1. Install Docker

Run the following commands on **Ubuntu** or **WSL** to install Docker Engine and Docker Compose:

```bash
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg]   https://download.docker.com/linux/ubuntu   $(lsb_release -cs) stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Check Docker installation
sudo docker --version

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify that Docker works
docker ps
```

---

## 2. Build and Run the Docker Image

Make sure your project directory contains:

```
.
├── Dockerfile
├── predict.py
├── pipeline.bin
├── pyproject.toml
└── uv.lock
```

Then build and run the container:

```bash
# Build the image
docker build -t predict:latest .

# Run the container
docker run -it --rm -p 9696:9696 predict:latest
```

Once the container is running, your FastAPI app will be available at:

```
http://localhost:9696/predict
```

---

## 3. Test the Prediction Endpoint

Run the test request using your local environment (with uv or Python):

```bash
uv run test_request.py
```

### Example Output

```json
{
  "probability_gt_50k": 0.8123,
  "threshold": 0.5,
  "predicted_label": ">50K"
}
```