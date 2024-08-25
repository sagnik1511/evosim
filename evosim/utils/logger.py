import logging


def get_logger():
    # Configure the logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%s",
    )

    return logging.getLogger()
