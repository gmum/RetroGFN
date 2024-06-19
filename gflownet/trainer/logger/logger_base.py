from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class LoggerBase(ABC):
    """
    The base class for loggers used in Trainer.
    """

    def __init__(self, logdir: str | Path):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        """
        Log the metrics.

        Args:
            metrics: a dictionary containing the metrics.
            prefix: a prefix to add to the metrics names.

        Returns:
            None
        """
        ...

    @abstractmethod
    def log_code(self, source_path: str | Path):
        """
        Log the code.

        Args:
            source_path: code source path.

        Returns:
            None
        """
        ...

    @abstractmethod
    def log_to_file(self, content: Any, name: str, type: str = "txt"):
        """
        Log anything to a file.

        Args:
            content: a content to log.
            name: file name.
            type: extension of the file.

        Returns:
            None
        """
        ...

    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """
        Log the configuration dictionary.

        Args:
            config: a configuration dictionary.

        Returns:
            None
        """
        ...

    @abstractmethod
    def close(self):
        """
        Close the logger. This method should be called at the end of the training.

        Returns:
            None
        """
        ...

    @abstractmethod
    def restart(self):
        """
        Restart the logger. This method should be called at the beginning of the training.

        Returns:
            None
        """
        ...
