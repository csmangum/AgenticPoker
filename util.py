import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def setup_logging(session_id=0):
    """
    Configure logging with UTF-8 encoding support and session management.

    Sets up a logging system that outputs to both console and file, with UTF-8 encoding.
    Creates a new log file for each session and configures the root logger with
    appropriate formatting.

    Args:
        session_id (int | str, optional): Unique identifier for this game session.
            If 0 or falsy, generates timestamp-based ID. Defaults to 0.

    Returns:
        None

    Side Effects:
        - Clears existing logging handlers
        - Creates new log file named 'poker_game.log'
        - Configures console and file output with UTF-8 encoding
        - Sets httpx logging level to WARNING
        - Logs session start information with timestamp
    """
    # Clear any existing handlers
    logging.getLogger().handlers = []

    # Generate session ID if not provided
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log filename with session ID
    log_filename = f"poker_game.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            # Console handler with UTF-8 encoding
            logging.StreamHandler(sys.stdout),
            # File handler with UTF-8 encoding and session-specific file
            logging.FileHandler(log_filename, encoding="utf-8", mode="w"),
        ],
    )

    # Silence httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Log session start with clear separator
    logging.info(f"\n{'='*70}")
    logging.info(f"New Poker Game Session Started - ID: {session_id}")
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*70}\n")


def clear_results_directory() -> None:
    """Clear previous game results while preserving ChromaDB connections."""
    results_dir = os.path.join(os.getcwd(), "results")
    chroma_dir = os.path.join(results_dir, "chroma_db")

    try:
        # Only remove non-ChromaDB files/directories
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if item != "chroma_db":  # Skip the ChromaDB directory
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        logger.info("Cleared previous game results (preserved ChromaDB)")
    except Exception as e:
        logger.error(f"Error clearing results directory: {str(e)}")


def load_agent_configs() -> Dict:
    """Load agent configurations from JSON file.

    Returns:
        Dict containing agent configurations or empty dict if file doesn't exist
    """
    config_path = Path("configs/agent_configs.json")
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading agent configs: {str(e)}")
        return {}


def save_agent_configs(configs: Dict) -> None:
    """Save agent configurations to JSON file.

    Args:
        configs: Dictionary of agent configurations to save
    """
    config_path = Path("configs/agent_configs.json")
    config_path.parent.mkdir(exist_ok=True)

    try:
        with open(config_path, "w") as f:
            json.dump(configs, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving agent configs: {str(e)}")


def ensure_directory_structure() -> None:
    """Ensure all required directories exist."""
    directories = ["logs", "results", "configs", "data"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
