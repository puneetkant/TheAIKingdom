# 5.8.4 Containerisation and Orchestration

Docker, docker-compose, Kubernetes, Helm, resource limits, health checks, autoscaling.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | K8s deployment YAML generator |
| `working_example2.py` | Dockerfile + compose generator + memory estimation |
| `working_example.ipynb` | Interactive: Dockerfile + memory bar chart |

## Quick Reference

```dockerfile
# Dockerfile for ML API
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-api
        image: ml-api:v1.0
        resources:
          requests: {memory: "512Mi", cpu: "250m"}
          limits:   {memory: "2Gi",  cpu: "1"}
        livenessProbe:
          httpGet: {path: /health, port: 8000}
```

## Key Concepts

| Tool | Purpose |
|------|---------|
| Docker | Package app + deps into image |
| docker-compose | Multi-container local dev |
| Kubernetes | Production orchestration |
| Helm | K8s package manager |

## Learning Resources
- [Docker for ML](https://docs.docker.com/)
- [Kubernetes basics](https://kubernetes.io/docs/tutorials/)

Explore this topic with a small practical project or coding exercise.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
