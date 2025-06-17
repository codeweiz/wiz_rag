import logging


# 日志装饰器
def log(cls):
    cls.logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
    return cls


# Mixin Log 基类
class LogMixin:
    logger: logging.Logger

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
