import logging
import os


def setup_logger(logger, logger_dir, stage="train"):
    os.makedirs(logger_dir, exist_ok=True)
    fh = logging.FileHandler(f"{logger_dir}/{stage}.log")

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    fh.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(fh)
