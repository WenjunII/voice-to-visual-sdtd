import json
import logging
import re
import sys
from dataclasses import fields
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4


_STANDARD_RECORD_FIELDS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)
_CONTEXT_FIELDS = {"session_id", "subsystem", "event"}


def _configured_secrets(config):
    return tuple(
        str(getattr(config, config_field.name))
        for config_field in fields(config)
        if config_field.metadata.get("secret")
        and config.is_secret_configured(getattr(config, config_field.name))
    )


class SecretRedactor:
    _credential_patterns = (
        re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
        re.compile(
            r"(?i)((?:api[_-]?key|authorization|access[_-]?token)"
            r"\s*[=:]\s*)[^\s,;]+"
        ),
    )

    def __init__(self, secrets=()):
        self.secrets = tuple(
            sorted(
                {str(secret) for secret in secrets if str(secret)},
                key=len,
                reverse=True,
            )
        )

    def text(self, value):
        redacted = str(value)
        for secret in self.secrets:
            redacted = redacted.replace(secret, "<redacted>")
        for pattern in self._credential_patterns:
            redacted = pattern.sub(r"\1<redacted>", redacted)
        return redacted

    def value(self, value):
        if isinstance(value, dict):
            return {
                self.text(key): self.value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self.value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self.text(value)


class SessionContextFilter(logging.Filter):
    def __init__(self, session_id, logger_prefix):
        super().__init__()
        self.session_id = session_id
        self.logger_prefix = logger_prefix

    def filter(self, record):
        record.session_id = self.session_id
        prefix = f"{self.logger_prefix}."
        record.subsystem = (
            record.name[len(prefix):]
            if record.name.startswith(prefix)
            else "runtime"
        )
        return True


class ConsoleFormatter(logging.Formatter):
    def __init__(self, redactor):
        super().__init__()
        self.redactor = redactor

    def format(self, record):
        timestamp = datetime.fromtimestamp(
            record.created,
            timezone.utc,
        ).astimezone().isoformat(timespec="seconds")
        message = self.redactor.text(record.getMessage()).strip()
        return (
            f"{timestamp} | {record.levelname:<7} | "
            f"{record.subsystem} | {message}"
        )


class JsonFormatter(logging.Formatter):
    def __init__(self, redactor):
        super().__init__()
        self.redactor = redactor

    def format(self, record):
        timestamp = datetime.fromtimestamp(
            record.created,
            timezone.utc,
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        payload = {
            "timestamp": timestamp,
            "level": record.levelname,
            "session_id": record.session_id,
            "subsystem": record.subsystem,
            "event": getattr(record, "event", "log"),
            "message": self.redactor.text(record.getMessage()).strip(),
        }
        for key, value in record.__dict__.items():
            if (
                key in _STANDARD_RECORD_FIELDS
                or key in _CONTEXT_FIELDS
                or key.startswith("_")
            ):
                continue
            payload[key] = self.redactor.value(value)
        if record.exc_info:
            payload["exception"] = self.redactor.text(
                self.formatException(record.exc_info)
            )
        return json.dumps(payload, ensure_ascii=False, default=str)


class RuntimeLogSession:
    def __init__(
        self,
        config,
        *,
        stream=None,
        session_id=None,
    ):
        self.config = config
        self.session_id = session_id or self._new_session_id()
        self.logger_name = f"voice_to_visual.session.{self.session_id}"
        self.path = None
        self.closed = False
        self._handlers = []

        self._base_logger = logging.getLogger(self.logger_name)
        self._base_logger.handlers.clear()
        self._base_logger.setLevel(logging.DEBUG)
        self._base_logger.propagate = False

        redactor = SecretRedactor(_configured_secrets(config))
        context_filter = SessionContextFilter(
            self.session_id,
            self.logger_name,
        )
        level = getattr(logging, config.runtime_log_level.upper())

        if config.runtime_log_console_enabled:
            console_handler = logging.StreamHandler(stream or sys.stdout)
            console_handler.setLevel(level)
            console_handler.addFilter(context_filter)
            console_handler.setFormatter(ConsoleFormatter(redactor))
            self._add_handler(console_handler)

        if config.runtime_log_file:
            try:
                self.path = Path(config.runtime_log_file).expanduser().resolve()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    self.path,
                    maxBytes=config.runtime_log_max_bytes,
                    backupCount=config.runtime_log_backup_count,
                    encoding="utf-8",
                    delay=False,
                )
                file_handler.setLevel(level)
                file_handler.addFilter(context_filter)
                file_handler.setFormatter(JsonFormatter(redactor))
                self._add_handler(file_handler)
            except OSError as exc:
                self.close()
                raise RuntimeError(
                    "Could not initialize RUNTIME_LOG_FILE="
                    f"{config.runtime_log_file}: {exc}"
                ) from exc

    def logger(self, subsystem):
        return self._base_logger.getChild(str(subsystem).strip() or "runtime")

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handler in tuple(self._handlers):
            self._base_logger.removeHandler(handler)
            handler.flush()
            handler.close()
        self._handlers.clear()

    def _add_handler(self, handler):
        self._base_logger.addHandler(handler)
        self._handlers.append(handler)

    @staticmethod
    def _new_session_id():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4().hex[:8]}"
