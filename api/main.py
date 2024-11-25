from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn
import os
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from model.gradient_descent import GradientDescentLinearRegression


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model():
    """
    Load the model with enhanced debugging and path resolution
    """
    try:
        # Register the custom class for pickle
        sys.modules["model.gradient_descent"] = sys.modules[
            GradientDescentLinearRegression.__module__
        ]

        # Get various possible paths
        current_file_dir = Path(__file__).parent.absolute()
        model_dir = current_file_dir / "model"

        # List all possible model locations
        possible_paths = [
            model_dir / "mental_health_prediction_model.pkl",
            current_file_dir / "mental_health_prediction_model.pkl",
        ]

        # Debug information
        logger.debug(f"Current file directory: {current_file_dir}")
        logger.debug(f"Model directory: {model_dir}")
        logger.debug("Checking following paths:")
        for path in possible_paths:
            logger.debug(f"- {path} (exists: {path.exists()})")

        # Try loading from each path
        for model_path in possible_paths:
            if model_path.exists():
                logger.info(f"Found model at: {model_path}")
                try:
                    with open(model_path, "rb") as file:
                        model_info = pickle.load(file)

                    # Validate model contents
                    required_keys = ["model", "scaler", "feature_columns"]
                    if all(key in model_info for key in required_keys):
                        logger.info("Model loaded successfully")
                        logger.debug(f"Model info keys: {list(model_info.keys())}")
                        return model_info
                    else:
                        logger.warning(
                            f"Model file at {model_path} missing required keys"
                        )
                        continue
                except Exception as e:
                    logger.error(f"Error reading model file at {model_path}: {str(e)}")
                    continue

        # If we get here, no valid model was found
        raise FileNotFoundError(
            f"Model file not found in any of the expected locations: {[str(p) for p in possible_paths]}"
        )

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Global variable for model info
model_info = None

app = FastAPI(
    title="Mental Health Prediction API",
    description="""
    This API provides mental health risk predictions based on lifestyle and health data.
    
    ## Features
    * Predicts anxiety and depression risk levels
    * Provides confidence scores for predictions
    * Identifies contributing risk factors
    * Validates input data ranges
    
    ## Usage
    Send a POST request to `/predict` with the required health metrics to get a prediction.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models with enhanced documentation
class HealthDataInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="User's age in years", example=30)
    sleep_quality: float = Field(
        ...,
        ge=0,
        le=10,
        description="Self-reported sleep quality score (0-10)",
        example=7.5,
    )
    daily_steps: int = Field(
        ..., ge=0, description="Number of steps walked per day", example=8000
    )
    calories_burned: float = Field(
        ..., ge=0, description="Total calories burned per day", example=2200.5
    )
    physical_activity_level: int = Field(
        ...,
        ge=0,
        le=2,
        description="Physical activity level (0: Low, 1: Medium, 2: High)",
        example=1,
    )
    heart_rate: int = Field(
        ...,
        ge=40,
        le=200,
        description="Resting heart rate in beats per minute",
        example=70,
    )
    social_interaction: int = Field(
        ...,
        ge=0,
        le=4,
        description="Social interaction level (0: None to 4: Very High)",
        example=3,
    )
    medication_usage: int = Field(
        ..., ge=0, le=1, description="Medication usage (0: Yes, 1: No)", example=1
    )
    sleep_duration: float = Field(
        ..., ge=0, le=24, description="Total sleep duration in hours", example=7.5
    )

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    anxiety_level: float = Field(..., description="Predicted anxiety level (0-1)")
    anxiety_risk: str = Field(..., description="Risk category for anxiety")
    anxiety_confidence: float = Field(
        ..., description="Confidence score for anxiety prediction"
    )
    depression_level: float = Field(..., description="Predicted depression level (0-1)")
    depression_risk: str = Field(..., description="Risk category for depression")
    depression_confidence: float = Field(
        ..., description="Confidence score for depression prediction"
    )
    contributing_factors: List[str] = Field(
        ..., description="List of identified risk factors"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="API response status")
    message: str = Field(..., description="Response message")
    data: Optional[PredictionResponse] = Field(None, description="Prediction results")


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@app.on_event("startup")
async def startup_event():
    """
    Startup event to load model when the application starts
    """
    global model_info
    try:
        logger.info("Starting model loading process")
        model_info = load_model()
        logger.info("Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        model_info = None


# Updating the model loading code with better error handling
def load_model():
    try:
        # Import the GradientDescentLinearRegression class
        # from model.gradient_descent import GradientDescentLinearRegression

        # Ensure the correct file path to the model
        model_path = os.path.join(
            os.path.dirname(__file__), "mental_health_prediction_model.pkl"
        )
        print(f"Loading model from: {model_path}")

        with open(model_path, "rb") as file:
            model_info = pickle.load(file)

        # Log model_info to check what was loaded
        print(f"Model loaded successfully: {model_info}")

        if not model_info or "model" not in model_info or "scaler" not in model_info:
            raise RuntimeError("Model or scaler is missing from the loaded model file.")

        return model_info

    except FileNotFoundError:
        raise RuntimeError(
            "Model file not found. Please ensure the model file exists in the correct location."
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Load the model at startup
try:
    model_info = load_model()
    print(f"Loaded model_info: {model_info}")
except Exception as e:
    print(f"Error during model loading: {str(e)}")
    model_info = None


def normalize_prediction(pred_value: float) -> tuple[str, float]:
    """Convert numerical prediction to risk level with confidence"""
    if pred_value < 0.33:
        return "Low Risk", (0.33 - pred_value) / 0.33
    elif pred_value < 0.66:
        return "Moderate Risk", 1 - abs(pred_value - 0.5) / 0.17
    else:
        return "High Risk", (pred_value - 0.66) / 0.34


def get_contributing_factors(data: Dict) -> List[str]:
    """Identify contributing risk factors from input data"""
    factors = []
    if data["sleep_quality"] < 6:
        factors.append("Poor sleep quality")
    if data["physical_activity_level"] == 0:
        factors.append("Low physical activity")
    if data["social_interaction"] < 2:
        factors.append("Limited social interaction")
    if data["daily_steps"] < 5000:
        factors.append("Insufficient daily movement")
    if data["sleep_duration"] < 6:
        factors.append("Insufficient sleep duration")
    return factors


@app.get(
    "/",
    response_model=Dict[str, str],
    tags=["Health"],
    summary="Root endpoint",
    description="Returns a welcome message to confirm the API is running.",
)
async def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A welcome message
    """
    return {"message": "Welcome to the Mental Health Prediction API"}


@app.post(
    "/predict",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Predict mental health risks",
    description="Analyzes health metrics to predict anxiety and depression risks.",
    response_description="Prediction results including risk levels and contributing factors",
)
async def predict_mental_health(data: HealthDataInput):
    """
    Predicts mental health risks based on provided health metrics.

    Args:
        data (HealthDataInput): Health metrics data

    Returns:
        HealthResponse: Prediction results

    Raises:
        HTTPException: If prediction fails
    """
    try:
        if model_info is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict()])
        df.rename(
            columns={
                "age": "Age",
                "sleep_quality": "Sleep Quality",
                "daily_steps": "Daily Steps",
                "calories_burned": "Calories Burned",
                "physical_activity_level": "Physical Activity Level",
                "heart_rate": "Heart Rate",
                "social_interaction": "Social Interaction",
                "medication_usage": "Medication Usage",
                "sleep_duration": "Sleep Duration",
            },
            inplace=True,
        )

        # Scale the features
        scaled_data = model_info["scaler"].transform(df[model_info["feature_columns"]])

        # Make prediction
        predictions = model_info["model"].predict(scaled_data)[0]
        if predictions is None:
            raise HTTPException(
                status_code=500, detail="Prediction failed. Model returned None."
            )

        # Get risk levels and confidence
        anxiety_risk, anxiety_conf = normalize_prediction(predictions[0])
        depression_risk, depression_conf = normalize_prediction(predictions[1])

        # Get contributing factors
        factors = get_contributing_factors(data.dict())

        prediction_result = {
            "anxiety_level": float(predictions[0]),
            "anxiety_risk": anxiety_risk,
            "anxiety_confidence": float(anxiety_conf),
            "depression_level": float(predictions[1]),
            "depression_risk": depression_risk,
            "depression_confidence": float(depression_conf),
            "contributing_factors": factors,
        }

        return HealthResponse(
            status="success",
            message="Prediction completed successfully",
            data=prediction_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Health check endpoint
@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns the health status of the API.",
    response_description="Health status information",
)
async def health_check():
    """
    Performs a health check of the API.

    Returns:
        dict: Health status information
    """
    return {"status": "healthy", "model_loaded": model_info is not None}


if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8001, reload=True, workers=4, log_level="info"
    )
