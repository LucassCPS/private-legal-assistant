import logging

LOGGER_NAME = "LEGAL_ASSISTANT_LOGGER"

class CustomFormatter(logging.Formatter):
    green = "\x1b[32m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    white = "\x1b[37m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: magenta + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def config_logger(logger_level=logging.DEBUG):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logger_level)
    logger.propagate = False # Importante para evitar logs duplicados

    # Verifica se o logger já tem handlers para não adicionar de novo
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logger_level)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    
    return logger