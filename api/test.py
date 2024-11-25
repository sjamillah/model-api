import requests

BASE_URL = "http://localhost:8001"


def test_root_endpoint():
    """Test the root endpoint"""
    response = requests.get(BASE_URL + "/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("Root endpoint test passed.")


def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(BASE_URL + "/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"
    print("Health check test passed.")


def test_predict_endpoint():
    payload = {
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

    response = requests.post(BASE_URL + "/predict", json=payload)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.json()}")
    assert response.status_code == 200
    assert response.json().get("status") == "success"


if __name__ == "__main__":
    print("Testing API endpoints...")
    test_root_endpoint()
    test_health_check()
    test_predict_endpoint()
    print("All tests passed!")
