"""
FastAPI Deployment Endpoint
Real-time portfolio allocation recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.dqn_agent import DQNAgent, QNetwork
from src.regime_detection.gmm_classifier import GMMRegimeDetector

# Initialize FastAPI
app = FastAPI(
    title="Deep RL Portfolio Allocation API",
    description="Real-time portfolio allocation recommendations using Deep RL",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
dqn_agent = None
regime_detector = None

# Request/Response models
class MarketState(BaseModel):
    """Market state input for prediction."""
    portfolio_weights: List[float] = Field(..., description="Current portfolio weights", min_items=4, max_items=4)
    recent_returns: List[float] = Field(..., description="Recent asset returns (last 5 days × 4 assets)", min_items=20, max_items=20)
    volatility: List[float] = Field(..., description="Current volatility for each asset", min_items=4, max_items=4)
    vix: float = Field(..., description="Current VIX level", ge=0)
    treasury_rate: float = Field(..., description="10Y Treasury rate (%)", ge=0)


class AllocationRecommendation(BaseModel):
    """Portfolio allocation recommendation."""
    action: int = Field(..., description="Recommended action: 0=Decrease, 1=Hold, 2=Increase")
    action_name: str = Field(..., description="Action description")
    confidence: float = Field(..., description="Confidence score (0-1)")
    current_regime: Optional[str] = Field(None, description="Detected market regime")
    q_values: List[float] = Field(..., description="Q-values for each action")


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


class RegimeInfo(BaseModel):
    """Market regime information."""
    regime_id: int
    regime_name: str
    confidence: float
    regime_stats: Dict[str, float]


@app.on_event("startup")
async def load_models():
    """Load trained models on startup."""
    global dqn_agent, regime_detector

    try:
        # Load DQN agent
        # Note: Update state_dim and action_dim based on your training
        state_dim = 34  # As per environment
        action_dim = 3  # Decrease/Hold/Increase

        dqn_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')

        # Try to load trained model
        try:
            dqn_agent.load("models/dqn_agent.pth")
            print("✓ DQN agent loaded successfully")
        except FileNotFoundError:
            print("⚠ Warning: DQN agent not trained yet. Using untrained agent.")

        # Load regime detector
        try:
            regime_detector = GMMRegimeDetector()
            regime_detector.load("models/gmm_regime_detector.pkl")
            print("✓ Regime detector loaded successfully")
        except FileNotFoundError:
            print("⚠ Warning: Regime detector not found. Regime detection disabled.")
            regime_detector = None

    except Exception as e:
        print(f"✗ Error loading models: {e}")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Deep RL Portfolio Allocation API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict - Get allocation recommendation",
            "regime": "/regime - Detect current market regime",
            "metrics": "/metrics - Calculate portfolio metrics",
            "health": "/health - API health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dqn_loaded": dqn_agent is not None,
        "regime_detector_loaded": regime_detector is not None
    }


@app.post("/predict", response_model=AllocationRecommendation)
async def predict_allocation(state: MarketState):
    """
    Get portfolio allocation recommendation.

    Args:
        state: Current market state

    Returns:
        Allocation recommendation with confidence
    """
    if dqn_agent is None:
        raise HTTPException(status_code=503, detail="DQN agent not loaded")

    try:
        # Construct state vector
        state_vector = []

        # Portfolio weights
        state_vector.extend(state.portfolio_weights)

        # Recent returns
        state_vector.extend(state.recent_returns)

        # Volatility
        state_vector.extend(state.volatility)

        # Regime (if detector available)
        if regime_detector is not None:
            # Simplified regime detection - in production, use proper features
            regime_one_hot = [0, 0, 0]  # Placeholder
            regime_one_hot[0] = 1  # Default to regime 0
        else:
            regime_one_hot = [0, 0, 0]

        state_vector.extend(regime_one_hot)

        # Macro indicators
        state_vector.append(state.vix)
        state_vector.append(state.treasury_rate)

        # Portfolio value (normalized)
        state_vector.append(1.0)

        # Convert to numpy
        state_array = np.array(state_vector, dtype=np.float32)

        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            q_values = dqn_agent.q_network(state_tensor).squeeze().numpy()

        # Get action
        action = int(q_values.argmax())

        # Action names
        action_names = {0: "Decrease Risky Allocation", 1: "Hold Current Allocation", 2: "Increase Risky Allocation"}

        # Calculate confidence (softmax of Q-values)
        exp_q = np.exp(q_values - q_values.max())
        confidence_scores = exp_q / exp_q.sum()
        confidence = float(confidence_scores[action])

        return AllocationRecommendation(
            action=action,
            action_name=action_names[action],
            confidence=confidence,
            current_regime="Bull" if regime_one_hot[0] == 1 else "Unknown",
            q_values=q_values.tolist()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/regime", response_model=RegimeInfo)
async def detect_regime(
    returns: List[float] = Field(..., description="Recent returns for regime detection"),
    vix: float = Field(..., description="Current VIX level")
):
    """
    Detect current market regime.

    Args:
        returns: Recent asset returns
        vix: VIX level

    Returns:
        Regime classification and statistics
    """
    if regime_detector is None:
        raise HTTPException(status_code=503, detail="Regime detector not loaded")

    try:
        # Create DataFrame for prediction
        returns_df = pd.DataFrame([returns], columns=['asset1', 'asset2', 'asset3', 'asset4'])
        vix_series = pd.Series([vix])

        # Predict regime
        regime_id = regime_detector.predict(returns_df, vix_series).iloc[0]

        regime_names = {0: 'Bull', 1: 'Bear', 2: 'Volatile'}
        regime_name = regime_names.get(int(regime_id), f'Regime {regime_id}')

        # Get probabilities
        probs = regime_detector.predict_proba(returns_df, vix_series)
        confidence = float(probs.iloc[0].max())

        return RegimeInfo(
            regime_id=int(regime_id),
            regime_name=regime_name,
            confidence=confidence,
            regime_stats={
                "bull_prob": float(probs.iloc[0][0]),
                "bear_prob": float(probs.iloc[0][1]),
                "volatile_prob": float(probs.iloc[0][2])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regime detection error: {str(e)}")


@app.post("/metrics", response_model=PerformanceMetrics)
async def calculate_metrics(
    portfolio_values: List[float] = Field(..., description="Historical portfolio values"),
    risk_free_rate: float = Field(0.02, description="Annual risk-free rate")
):
    """
    Calculate portfolio performance metrics.

    Args:
        portfolio_values: Historical portfolio values
        risk_free_rate: Annual risk-free rate

    Returns:
        Performance metrics
    """
    try:
        values = np.array(portfolio_values)

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Returns
        returns = np.diff(values) / values[:-1]

        # Sharpe ratio
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        return PerformanceMetrics(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics calculation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
