"""
Experiment Tracking Integration

Supports multiple backends:
- MLflow
- Weights & Biases (wandb)
- TensorBoard

Provides unified interface for logging experiments.
"""

import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface."""

    def __init__(
        self,
        backend: str = 'mlflow',
        experiment_name: str = 'portfolio_allocation',
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize experiment tracker.

        Args:
            backend: Tracking backend ('mlflow', 'wandb', 'tensorboard')
            experiment_name: Name of experiment
            run_name: Name of run (auto-generated if None)
            tracking_uri: URI for tracking server
            tags: Additional tags for run
        """
        self.backend = backend.lower()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize tracking backend."""

        if self.backend == 'mlflow':
            self._init_mlflow()
        elif self.backend == 'wandb':
            self._init_wandb()
        elif self.backend == 'tensorboard':
            self._init_tensorboard()
        else:
            logger.warning(f"Unknown backend '{self.backend}'. Using dummy tracker.")
            self.backend = 'dummy'

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            else:
                # Use local directory
                tracking_dir = Path('mlruns')
                tracking_dir.mkdir(exist_ok=True)
                mlflow.set_tracking_uri(f'file://{tracking_dir.absolute()}')

            mlflow.set_experiment(self.experiment_name)

            self.run = mlflow.start_run(run_name=self.run_name)

            # Log tags
            for key, value in self.tags.items():
                mlflow.set_tag(key, value)

            logger.info(f"MLflow tracking initialized")
            logger.info(f"  Experiment: {self.experiment_name}")
            logger.info(f"  Run ID: {self.run.info.run_id}")

        except ImportError:
            logger.error("MLflow not installed. Install with: pip install mlflow")
            self.backend = 'dummy'

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb

            wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                tags=list(self.tags.values()),
                config=self.tags
            )

            logger.info(f"Weights & Biases tracking initialized")
            logger.info(f"  Project: {self.experiment_name}")

        except ImportError:
            logger.error("wandb not installed. Install with: pip install wandb")
            self.backend = 'dummy'

    def _init_tensorboard(self):
        """Initialize TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path('runs') / self.experiment_name
            if self.run_name:
                log_dir = log_dir / self.run_name

            self.writer = SummaryWriter(log_dir=str(log_dir))

            logger.info(f"TensorBoard tracking initialized")
            logger.info(f"  Log dir: {log_dir}")

        except ImportError:
            logger.error("TensorBoard not installed. Install with: pip install tensorboard")
            self.backend = 'dummy'

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""

        if self.backend == 'mlflow':
            import mlflow
            mlflow.log_params(params)

        elif self.backend == 'wandb':
            import wandb
            wandb.config.update(params)

        elif self.backend == 'tensorboard':
            # TensorBoard doesn't have built-in param logging
            # We can log as text
            pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""

        if self.backend == 'mlflow':
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

        elif self.backend == 'wandb':
            import wandb
            wandb.log(metrics, step=step)

        elif self.backend == 'tensorboard':
            if step is None:
                step = 0
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        """Log artifact (file)."""

        if self.backend == 'mlflow':
            import mlflow
            mlflow.log_artifact(filepath, artifact_path=artifact_path)

        elif self.backend == 'wandb':
            import wandb
            wandb.save(filepath)

    def log_model(self, model, model_name: str = 'model'):
        """Log model."""

        if self.backend == 'mlflow':
            import mlflow
            import mlflow.pytorch

            # Save as PyTorch model
            mlflow.pytorch.log_model(model, model_name)

        elif self.backend == 'wandb':
            import wandb
            # wandb can log model artifacts
            pass

    def finish(self):
        """Finish tracking run."""

        if self.backend == 'mlflow':
            import mlflow
            mlflow.end_run()
            logger.info("MLflow run finished")

        elif self.backend == 'wandb':
            import wandb
            wandb.finish()
            logger.info("Wandb run finished")

        elif self.backend == 'tensorboard':
            self.writer.close()
            logger.info("TensorBoard writer closed")


class DummyTracker:
    """Dummy tracker when no backend is available."""

    def log_params(self, params: Dict[str, Any]):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        pass

    def log_model(self, model, model_name: str = 'model'):
        pass

    def finish(self):
        pass


def get_tracker(
    backend: str = 'mlflow',
    experiment_name: str = 'default',
    **kwargs
) -> ExperimentTracker:
    """
    Get experiment tracker instance.

    Args:
        backend: Tracking backend
        experiment_name: Experiment name
        **kwargs: Additional arguments

    Returns:
        tracker: ExperimentTracker instance
    """
    try:
        return ExperimentTracker(
            backend=backend,
            experiment_name=experiment_name,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to initialize tracker: {e}")
        return DummyTracker()


if __name__ == '__main__':
    # Example usage

    # MLflow example
    tracker = ExperimentTracker(
        backend='mlflow',
        experiment_name='test_experiment',
        run_name='test_run',
        tags={'algorithm': 'SAC', 'version': '1.0'}
    )

    # Log hyperparameters
    tracker.log_params({
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'batch_size': 256
    })

    # Log metrics
    for step in range(100):
        tracker.log_metrics({
            'reward': step * 0.1,
            'loss': 1.0 / (step + 1)
        }, step=step)

    # Finish
    tracker.finish()

    print("Experiment tracking demo complete!")
