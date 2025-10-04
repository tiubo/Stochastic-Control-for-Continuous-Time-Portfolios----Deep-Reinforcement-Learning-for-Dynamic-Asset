from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.dqn_agent import DQNAgent
from src.environments.portfolio_env import PortfolioEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Allocation API",
    description="Deep RL-based portfolio optimization using DQN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent = None
env = None
data = None

# Pydantic models
class StateInput(BaseModel):
    state: List[float]

class AllocationResponse(BaseModel):
    allocation: Dict[str, float]
    sharpe_ratio: float
    model: str

@app.on_event("startup")
async def load_model():
    """Load model and data on startup"""
    global agent, env, data

    try:
        # Load historical data
        data = pd.read_csv('data/processed/dataset_with_regimes.csv',
                          index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(data)} days of market data")

        # Create environment
        env = PortfolioEnv(data=data, action_type='discrete')
        logger.info(f"Environment created: state_dim={env.observation_space.shape[0]}")

        # Load DQN agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            device='cpu'
        )
        agent.load('models/dqn_trained_ep1000.pth')
        logger.info("DQN agent loaded successfully")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Portfolio Allocation API",
        "status": "running",
        "model": "DQN",
        "sharpe_ratio": 2.293,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if agent is None or env is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=AllocationResponse)
async def predict_allocation(input_data: StateInput):
    """Predict optimal portfolio allocation from state vector"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get action from agent
        state = np.array(input_data.state)
        action = agent.select_action(state, epsilon=0)
        weights = env.discrete_actions[action]

        # Return allocation
        allocation = {
            "SPY": float(weights[0]),
            "TLT": float(weights[1]),
            "GLD": float(weights[2]),
            "BTC": float(weights[3])
        }

        return AllocationResponse(
            allocation=allocation,
            sharpe_ratio=2.293,
            model="DQN"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/allocate")
async def get_current_allocation():
    """Get current recommended allocation"""
    if env is None or agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get latest state
        state, _ = env.reset()
        action = agent.select_action(state, epsilon=0)
        weights = env.discrete_actions[action]

        return {
            "allocation": {
                "SPY": float(weights[0]),
                "TLT": float(weights[1]),
                "GLD": float(weights[2]),
                "BTC": float(weights[3])
            },
            "timestamp": pd.Timestamp.now().isoformat(),
            "model": "DQN",
            "sharpe_ratio": 2.293
        }

    except Exception as e:
        logger.error(f"Allocation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    return {
        "model": "DQN",
        "total_return": "247.66%",
        "sharpe_ratio": 2.293,
        "sortino_ratio": 3.541,
        "max_drawdown": "20.37%",
        "calmar_ratio": 12.16,
        "training_episodes": 1000,
        "test_period": "2022-12-14 to 2024-12-31"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
