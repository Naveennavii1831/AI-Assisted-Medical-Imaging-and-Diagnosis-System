
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "educational_prototype"
        assert data["clinical_use"] is False


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["clinical_use"] is False


def test_review_history_endpoint():
    with TestClient(app) as client:
        response = client.get("/reviews?limit=5")

        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["reviews"], list)
        assert data["count"] <= 5


def test_approve_review_and_retrieve_it():
    review_id = f"test-approve-{uuid4().hex}"

    payload = {
        "review_id": review_id,
        "filename": "test_approved_image.png",
        "decision": "approved",
        "reviewer": "Automated Test",
        "comments": "Automated approval test",
    }

    with TestClient(app) as client:
        response = client.post("/reviews", json=payload)

        assert response.status_code == 201
        assert response.json()["message"] == "Review saved successfully."

        saved_review = response.json()["review"]

        assert saved_review["review_id"] == review_id
        assert saved_review["decision"] == "approved"
        assert saved_review["reviewer"] == "Automated Test"
        assert saved_review["comments"] == "Automated approval test"

        retrieval = client.get(f"/reviews/{review_id}")

        assert retrieval.status_code == 200
        assert retrieval.json()["review_id"] == review_id
        assert retrieval.json()["decision"] == "approved"


def test_reject_review():
    review_id = f"test-reject-{uuid4().hex}"

    payload = {
        "review_id": review_id,
        "filename": "test_rejected_image.png",
        "decision": "rejected",
        "reviewer": "Automated Test",
        "comments": "Automated rejection test",
    }

    with TestClient(app) as client:
        response = client.post("/reviews", json=payload)

        assert response.status_code == 201
        assert response.json()["message"] == "Review saved successfully."

        saved_review = response.json()["review"]

        assert saved_review["review_id"] == review_id
        assert saved_review["decision"] == "rejected"
        assert saved_review["reviewer"] == "Automated Test"


def test_invalid_review_decision():
    payload = {
        "review_id": f"test-invalid-{uuid4().hex}",
        "filename": "test_image.png",
        "decision": "pending",
        "reviewer": "Automated Test",
        "comments": "Invalid decision test",
    }

    with TestClient(app) as client:
        response = client.post("/reviews", json=payload)

        assert response.status_code == 422


def test_review_not_found():
    review_id = f"missing-review-{uuid4().hex}"

    with TestClient(app) as client:
        response = client.get(f"/reviews/{review_id}")

        assert response.status_code == 404
