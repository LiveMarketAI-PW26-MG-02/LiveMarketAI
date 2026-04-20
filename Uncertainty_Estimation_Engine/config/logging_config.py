"""Logging configuration for the Uncertainty Estimation Engine."""
import logging
import logging.config
import yaml
from pathlib import Path


def setup_logging(config_path: str = "config/config.yaml") -> None:
    """Configure logging from YAML config."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        log_cfg = cfg.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        fmt = log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s")
        log_file = log_cfg.get("file", "uncertainty_engine.log")
        Path(log_file).parent.mkdir(exist_ok=True)
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)
