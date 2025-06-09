import logging

logger = logging.getLogger("mamba_ts_forecasting")
logger.setLevel(logging.INFO)

def configure_logger(log_path=None):
    # Avoid adding multiple handlers to the same logger
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Stream (console) handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Prevent logs from being passed to the root logger
    logger.propagate = False

    return logger