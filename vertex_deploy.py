# NOTE: Requires GCP billing enabled + google-cloud-aiplatform installed.
# This script is not executed in the current deployment (Streamlit Cloud).
# Documents the intended Vertex AI migration path for production deployment.

"""
vertex_deploy.py — uploads model to GCP and deploys to Vertex AI Endpoint.
Run ONCE after enabling GCP billing:
  python vertex_deploy.py
"""

import os
from google.cloud import aiplatform

PROJECT_ID   = "handy-droplet-459914-u3"
LOCATION     = "us-central1"
BUCKET_NAME  = "fifa-predictor-handy"
MODEL_DIR    = f"gs://{BUCKET_NAME}/model/"
DISPLAY_NAME = "fifa-wc-2026-predictor"

def deploy():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    print("Step 1: Registering model in Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name=DISPLAY_NAME,
        artifact_uri=MODEL_DIR,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    )
    print(f"Model registered: {model.resource_name}")

    print("Step 2: Deploying to Vertex AI Endpoint...")
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )
    print(f"Endpoint ID: {endpoint.name}")
    print("Set these in Streamlit Cloud secrets:")
    print(f"  USE_VERTEX_AI=true")
    print(f"  GCP_PROJECT_ID={PROJECT_ID}")
    print(f"  VERTEX_ENDPOINT_ID={endpoint.name}")
    print(f"  VERTEX_LOCATION={LOCATION}")

if __name__ == "__main__":
    deploy()