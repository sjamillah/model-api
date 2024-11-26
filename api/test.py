import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming the main file is named main.py
import json

# Create a test client
client = TestClient(app)

# Sample valid input data for testing
VALID_INPUT_DATA = {
    "age": 30,
    "sleep_quality": 7.5,
    "daily_steps": 8000,
    "calories_burned": 2200.5,
    "physical_activity_level": 1,
    "heart_rate": 70,
    "social_interaction": 3,
    "medication_usage": 1,
    "sleep_duration": 7.5,
}


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome to the Mental Health Prediction API" in response.json()["message"]


def test_health_check_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input data"""
    response = client.post("/predict", json=VALID_INPUT_DATA)

    # Check response status
    assert response.status_code == 200

    # Check response structure
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data

    # Check prediction data
    prediction = data["data"]
    assert "anxiety_level" in prediction
    assert "anxiety_risk" in prediction
    assert "anxiety_confidence" in prediction
    assert "depression_level" in prediction
    assert "depression_risk" in prediction
    assert "depression_confidence" in prediction
    assert "contributing_factors" in prediction

    # Validate value ranges
    assert 0 <= prediction["anxiety_level"] <= 1
    assert prediction["anxiety_risk"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert 0 <= prediction["anxiety_confidence"] <= 1

    assert 0 <= prediction["depression_level"] <= 1
    assert prediction["depression_risk"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert 0 <= prediction["depression_confidence"] <= 1


def test_predict_endpoint_invalid_inputs():
    """Test prediction endpoint with various invalid inputs"""
    # Test input below minimum age
    invalid_age_data = VALID_INPUT_DATA.copy()
    invalid_age_data["age"] = 10
    response = client.post("/predict", json=invalid_age_data)
    assert response.status_code == 422

    # Test invalid sleep quality
    invalid_sleep_data = VALID_INPUT_DATA.copy()
    invalid_sleep_data["sleep_quality"] = 11
    response = client.post("/predict", json=invalid_sleep_data)
    assert response.status_code == 422

    # Test negative daily steps
    invalid_steps_data = VALID_INPUT_DATA.copy()
    invalid_steps_data["daily_steps"] = -100
    response = client.post("/predict", json=invalid_steps_data)
    assert response.status_code == 422


def test_predict_endpoint_boundary_inputs():
    """Test prediction endpoint with boundary condition inputs"""
    # Minimum valid age
    min_age_data = VALID_INPUT_DATA.copy()
    min_age_data["age"] = 18
    response = client.post("/predict", json=min_age_data)
    assert response.status_code == 200

    # Maximum valid age
    max_age_data = VALID_INPUT_DATA.copy()
    max_age_data["age"] = 100
    response = client.post("/predict", json=max_age_data)
    assert response.status_code == 200


def test_documentation_endpoints():
    """Test API documentation endpoints"""
    # Test Swagger UI
    swagger_response = client.get("/docs")
    assert swagger_response.status_code == 200

    # Test ReDoc
    redoc_response = client.get("/redoc")
    assert redoc_response.status_code == 200


def test_input_data_type_validations():
    """Test input data type validations"""
    # Test string instead of integer for age
    invalid_type_data = VALID_INPUT_DATA.copy()
    invalid_type_data["age"] = "thirty"
    response = client.post("/predict", json=invalid_type_data)
    assert response.status_code == 422


def test_openapi_schema():
    """Test OpenAPI schema is accessible"""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200

    # Verify basic schema structure
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
