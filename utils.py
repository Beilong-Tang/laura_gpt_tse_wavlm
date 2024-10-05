import datetime
import os
import logging


def init(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])


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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None
