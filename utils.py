import datetime
import os
import logging

def setup_logger(args):
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = args.log
    print(f"logging dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/{now}.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    logger.info("logger initialized")
    return logger