import datetime
import os
import logging

from argparse import Namespace


def init(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])


def setup_logger(args: Namespace, rank: int):
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(args.log, os.path.basename(args.config).replace(".yaml", ""))
    print(f"logging dir: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(name)s,%(levelname)s,%(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/{now}.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    logger.info("logger initialized")
    return Logger(logger, rank)


class Logger:
    def __init__(self, log: logging.Logger, rank: int):
        self.log = log
        self.rank = rank

    def info(self, msg: str):
        if self.rank == 0:
            self.log.info(msg)

    def debug(self, msg: str):
        if self.rank == 0:
            self.log.debug(msg)
        pass

    def warning(self, msg: str):
        self.log.warning(f"rank {self.rank} - {msg}")
        pass

    def error(self, msg: str):
        self.log.error(f"rank {self.rank} - {msg}")
        pass

    def critical(self, msg: str):
        self.log.critical(f"rank {self.rank} - {msg}")

        pass


class AttrDict(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None
