import logging

from wiz_rag.config import toml_config
from wiz_rag.utils.logger import LogMixin


# 使用装饰器为类自动注入 logger
class MyClass(LogMixin):
    def foo(self):
        self.logger.info("Hello, World!")


if __name__ == "__main__":
    my_class = MyClass()
    my_class.foo()
    logging.info(toml_config.llm.provider)
    logging.info(toml_config.llm.model_name)
    logging.info(toml_config.llm.api_key)
