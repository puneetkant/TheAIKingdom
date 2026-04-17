# 5.8.7 Cloud Platforms for ML

AWS SageMaker, GCP Vertex AI, Azure ML — managed training, serving, pipelines, cost optimisation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | SageMaker / Vertex AI SDK examples |
| `working_example2.py` | Cost estimator: training + inference across instance types |
| `working_example.ipynb` | Interactive: cloud cost curves |

## Quick Reference

```python
# AWS SageMaker training job
import sagemaker
estimator = sagemaker.sklearn.SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
)
estimator.fit({"train": s3_train_uri})

# GCP Vertex AI
from google.cloud import aiplatform
job = aiplatform.CustomTrainingJob(
    display_name="train-job",
    script_path="train.py",
    container_uri="gcr.io/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest",
)
job.run(machine_type="n1-standard-4")

# Spot/Preemptible instances: 60-90% cheaper
# Use checkpointing to resume interrupted runs
```

## Platform Comparison

| Platform | Managed Training | Serving | Pipelines |
|----------|-----------------|---------|----------|
| AWS SageMaker | ✓ | ✓ (endpoints) | ✓ (Pipelines) |
| GCP Vertex AI | ✓ | ✓ (endpoints) | ✓ (Pipelines) |
| Azure ML | ✓ | ✓ (ACI/AKS) | ✓ (Designer) |
| Databricks | ✓ (MLflow) | ✓ | ✓ |

## Learning Resources
- [SageMaker docs](https://docs.aws.amazon.com/sagemaker/)
- [Vertex AI docs](https://cloud.google.com/vertex-ai/docs)
- [Azure ML docs](https://learn.microsoft.com/en-us/azure/machine-learning/)

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
