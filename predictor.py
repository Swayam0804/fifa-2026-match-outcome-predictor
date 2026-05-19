"""
predictor.py — unified prediction interface.

Set environment variable USE_VERTEX_AI=true to route predictions
through a Vertex AI Endpoint instead of the local joblib model.

NOTE: Vertex AI deployment requires GCP billing enabled.
This script defaults to local mode (USE_VERTEX_AI=false).
"""

import os
import numpy as np

USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "false").lower() == "true"

def predict_local(features: list, model, le) -> dict:
    X = np.array(features).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    return {cls: float(prob) for cls, prob in zip(le.classes_, probs)}

def predict_vertex(features: list) -> dict:
    from google.cloud import aiplatform
    project_id  = os.getenv("GCP_PROJECT_ID")
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID")
    location    = os.getenv("VERTEX_LOCATION", "us-central1")
    if not project_id or not endpoint_id:
        raise EnvironmentError("Missing GCP_PROJECT_ID or VERTEX_ENDPOINT_ID.")
    aiplatform.init(project=project_id, location=location)
    endpoint = aiplatform.Endpoint(endpoint_id)
    response = endpoint.predict(instances=[features])
    classes = ["Away Win", "Draw", "Home Win"]
    return {cls: float(prob) for cls, prob in zip(classes, response.predictions[0])}

def predict(features: list, model=None, le=None) -> dict:
    if USE_VERTEX_AI:
        return predict_vertex(features)
    else:
        if model is None or le is None:
            raise ValueError("model and le required in local mode.")
        return predict_local(features, model, le)