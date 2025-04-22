import logging
import os.path
import inspect
from datetime import datetime

class MyFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        if not datefmt:
            return super().formatTime(record, datefmt=datefmt)

        return datetime.fromtimestamp(record.created).strftime(datefmt)


def create_logger(log_name: str, local_rank: int, save: bool = True):
    log_name = f"{log_name}_rank_{local_rank}.log"
    formatter = MyFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                            "%Y-%m-%d %H:%M:%S.%f %Z")
    file_handler = logging.FileHandler(log_name)
    logging.basicConfig(filename=log_name,
                        format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
                        filemode='w',
                        level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(log_name)
    logger.addHandler(file_handler)
    # logger.disabled = True
    # if not save:
    #     return logger
    # logger.setLevel(level=logging.DEBUG)
    # file_handler = logging.FileHandler(f"{log_name}.log", mode='w')
    # file_handler.setLevel(level=logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.ERROR)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)
    return logger

class Logger:
    def __init__(self, name: str, dump_interval: int = 0):
        self.name = name
        self.dump_interval = dump_interval
        self.msg_buff = []
        self.file_name = f'{name}.log'
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dump__()

    def __dump__(self):
        if not self.msg_buff:
            return
        self.file = open(self.file_name, 'a')
        self.file.writelines(self.msg_buff)
        self.msg_buff = []
        self.file.close()

    def add(self, msg, display_method=None):
        frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(frame).filename
        lineno = inspect.getframeinfo(frame).lineno
        formatted_datetime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        msg = f"{formatted_datetime} {filename}:{lineno} -- {msg}\n"
        if display_method:
            display_method(msg)
        self.msg_buff.append(msg)
        if len(self.msg_buff) >= self.dump_interval:
            self.__dump__()


if __name__ == "__main__":
    logger = Logger('test', dump_interval=10)
    for i in range(30):
        logger.add(f"hello_{i}")

