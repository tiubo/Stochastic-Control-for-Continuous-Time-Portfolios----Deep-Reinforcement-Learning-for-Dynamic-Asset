"""
Python-based deployment script for Deep RL Portfolio Allocation
Replaces docker-compose.yml with pure Python deployment
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from typing import Optional, List
import argparse


class DeploymentManager:
    """Manages deployment of the portfolio allocation API."""

    def __init__(
        self,
        model_path: str = "models/dqn_trained_ep1000.pth",
        data_path: str = "data/processed/dataset_with_regimes.csv",
        port: int = 8000,
        host: str = "0.0.0.0",
        log_level: str = "INFO",
        use_docker: bool = False
    ):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.port = port
        self.host = host
        self.log_level = log_level
        self.use_docker = use_docker
        self.process: Optional[subprocess.Popen] = None

    def validate_paths(self) -> bool:
        """Validate that required files exist."""
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            return False

        if not self.data_path.exists():
            print(f"❌ Data not found: {self.data_path}")
            return False

        print("✅ All required files found")
        return True

    def check_port_available(self) -> bool:
        """Check if the port is available."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((self.host, self.port))
            sock.close()
            return True
        except OSError:
            return False

    def deploy_docker(self) -> bool:
        """Deploy using Docker."""
        print("🐳 Deploying with Docker...")

        # Check if Docker is installed
        try:
            subprocess.run(
                ["docker", "--version"],
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker not installed. Please install Docker first.")
            return False

        # Build Docker image
        print("📦 Building Docker image...")
        build_cmd = ["docker", "build", "-t", "portfolio-api", "."]

        try:
            subprocess.run(build_cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Docker build failed")
            return False

        # Run Docker container
        print("🚀 Starting Docker container...")
        run_cmd = [
            "docker", "run", "-d",
            "--name", "portfolio-api",
            "-p", f"{self.port}:8000",
            "-v", f"{self.model_path.parent.absolute()}:/app/models:ro",
            "-v", f"{self.data_path.parent.absolute()}:/app/data:ro",
            "-v", f"{Path('logs').absolute()}:/app/logs",
            "-e", f"MODEL_PATH=/app/models/{self.model_path.name}",
            "-e", f"DATA_PATH=/app/data/{self.data_path.name}",
            "-e", f"LOG_LEVEL={self.log_level}",
            "--restart", "unless-stopped",
            "portfolio-api"
        ]

        try:
            subprocess.run(run_cmd, check=True)
            print(f"✅ Container started on http://{self.host}:{self.port}")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start container")
            return False

    def deploy_local(self) -> bool:
        """Deploy locally using uvicorn."""
        print("🚀 Deploying locally with Uvicorn...")

        # Set environment variables
        os.environ["MODEL_PATH"] = str(self.model_path.absolute())
        os.environ["DATA_PATH"] = str(self.data_path.absolute())
        os.environ["LOG_LEVEL"] = self.log_level

        # Change to deployment directory
        api_path = Path("src/deployment/api.py")
        if not api_path.exists():
            print(f"❌ API file not found: {api_path}")
            return False

        # Start uvicorn
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.deployment.api:app",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", self.log_level.lower()
        ]

        print(f"📡 Starting API server on http://{self.host}:{self.port}")
        print(f"   Model: {self.model_path}")
        print(f"   Data: {self.data_path}")

        try:
            self.process = subprocess.Popen(cmd)
            time.sleep(2)  # Give server time to start

            if self.process.poll() is None:
                print(f"\n✅ API server running (PID: {self.process.pid})")
                print(f"   Health check: http://{self.host}:{self.port}/health")
                print(f"   Docs: http://{self.host}:{self.port}/docs")
                print("\nPress Ctrl+C to stop the server")
                return True
            else:
                print("❌ Server failed to start")
                return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def deploy(self) -> bool:
        """Deploy the API."""
        if not self.validate_paths():
            return False

        if not self.check_port_available():
            print(f"❌ Port {self.port} is already in use")
            return False

        if self.use_docker:
            return self.deploy_docker()
        else:
            return self.deploy_local()

    def stop(self):
        """Stop the deployment."""
        if self.use_docker:
            print("🛑 Stopping Docker container...")
            subprocess.run(["docker", "stop", "portfolio-api"])
            subprocess.run(["docker", "rm", "portfolio-api"])
        elif self.process:
            print("\n🛑 Stopping API server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("✅ Server stopped")

    def wait(self):
        """Wait for the process to complete."""
        if self.process:
            try:
                self.process.wait()
            except KeyboardInterrupt:
                self.stop()


class StreamlitDeployment:
    """Deploy Streamlit dashboard."""

    def __init__(self, port: int = 8501):
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    def deploy(self) -> bool:
        """Deploy Streamlit dashboard."""
        print("📊 Starting Streamlit dashboard...")

        dashboard_path = Path("app/dashboard.py")
        if not dashboard_path.exists():
            print(f"❌ Dashboard not found: {dashboard_path}")
            return False

        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(self.port),
            "--server.address", "0.0.0.0"
        ]

        try:
            self.process = subprocess.Popen(cmd)
            time.sleep(2)

            if self.process.poll() is None:
                print(f"\n✅ Dashboard running on http://localhost:{self.port}")
                print("Press Ctrl+C to stop")
                return True
            else:
                print("❌ Dashboard failed to start")
                return False

        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False

    def stop(self):
        """Stop the dashboard."""
        if self.process:
            print("\n🛑 Stopping dashboard...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("✅ Dashboard stopped")

    def wait(self):
        """Wait for the process to complete."""
        if self.process:
            try:
                self.process.wait()
            except KeyboardInterrupt:
                self.stop()


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Deploy Deep RL Portfolio Allocation API"
    )
    parser.add_argument(
        "--service",
        choices=["api", "dashboard", "all"],
        default="all",
        help="Service to deploy"
    )
    parser.add_argument(
        "--model",
        default="models/dqn_trained_ep1000.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        default="data/processed/dataset_with_regimes.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API port"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Dashboard port"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker for deployment"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Create deployment managers
    api_manager = None
    dashboard_manager = None

    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n🛑 Shutting down...")
        if api_manager:
            api_manager.stop()
        if dashboard_manager:
            dashboard_manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Deploy API
        if args.service in ["api", "all"]:
            api_manager = DeploymentManager(
                model_path=args.model,
                data_path=args.data,
                port=args.api_port,
                log_level=args.log_level,
                use_docker=args.docker
            )

            if not api_manager.deploy():
                return 1

        # Deploy Dashboard
        if args.service in ["dashboard", "all"]:
            dashboard_manager = StreamlitDeployment(port=args.dashboard_port)

            if not dashboard_manager.deploy():
                if api_manager:
                    api_manager.stop()
                return 1

        # Wait for processes
        print("\n" + "="*60)
        print("🎉 Deployment complete!")
        print("="*60)

        if args.service in ["api", "all"]:
            print(f"\n📡 API: http://localhost:{args.api_port}")
            print(f"   Health: http://localhost:{args.api_port}/health")
            print(f"   Docs: http://localhost:{args.api_port}/docs")

        if args.service in ["dashboard", "all"]:
            print(f"\n📊 Dashboard: http://localhost:{args.dashboard_port}")

        print("\n" + "="*60)
        print("Press Ctrl+C to stop all services")
        print("="*60 + "\n")

        # Wait
        if api_manager:
            api_manager.wait()
        if dashboard_manager:
            dashboard_manager.wait()

    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        if api_manager:
            api_manager.stop()
        if dashboard_manager:
            dashboard_manager.stop()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
