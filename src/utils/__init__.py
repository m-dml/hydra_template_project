from dataclasses import dataclass


@dataclass
class LogLevelAbstractClass:
    log_level: str = "INFO"


LOG_LEVEL = LogLevelAbstractClass()
