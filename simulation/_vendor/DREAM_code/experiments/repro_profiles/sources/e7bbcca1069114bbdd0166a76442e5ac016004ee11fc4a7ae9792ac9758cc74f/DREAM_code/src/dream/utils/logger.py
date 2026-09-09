from typing import Optional

_COLOR_CODES = {
    "grey": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
}


def color_text(text: object, color: Optional[str] = None) -> str:
    """Return text with terminal color applied."""
    text = str(text)
    code = _COLOR_CODES.get(color or "")
    if code is None:
        return text
    return f"\033[{code}m{text}\033[0m"


def format_bool(
    value: Optional[bool],
    true_text: str = "True",
    false_text: str = "False",
    none_text: str = "None",
) -> str:
    """Format a boolean-like judgement with a consistent terminal color."""
    if value is True:
        return color_text(true_text, "green")
    if value is False:
        return color_text(false_text, "red")
    return color_text(none_text, "yellow")


class Logger:
    def __init__(self, name: str, hide_info: bool = False, hide_debug: bool = True) -> None:
        self.name = name
        self._hide_info = hide_info
        self._hide_debug = hide_debug

    def hide_info(self) -> None:
        self._hide_info = True

    def hide_debug(self) -> None:
        self._hide_debug = True

    def show_info(self) -> None:
        self._hide_info = False

    def show_debug(self) -> None:
        self._hide_debug = False

    def _flatten(self, args: tuple) -> str:
        """Flatten a tuple of arguments into a string joined by spaces.

        Args:
            args (tuple): Tuple of arguments to flatten.

        Returns:
            str: Flattened string.
        """
        text = " ".join([str(arg) for arg in args])
        if self.name is not None:
            text = f"[{self.name}] {text}"
        return text

    def error(self, *args) -> None:
        text = self._flatten(args)
        print(color_text(text, "red"))

    def info(self, *args) -> None:
        if not self._hide_info:
            text = self._flatten(args)
            print(color_text(text, "white"))

    def debug(self, *args) -> None:
        if not self._hide_debug:
            text = self._flatten(args)
            print(color_text(text, "white"))

    def warning(self, *args) -> None:
        text = self._flatten(args)
        print(color_text(text, "yellow"))

    def alert(self, *args) -> None:
        text = self._flatten(args)
        print(color_text(text, "green"))


_default_logger = Logger(None)


def error(*args) -> None:
    _default_logger.error(*args)


def info(*args) -> None:
    _default_logger.info(*args)


def warning(*args) -> None:
    _default_logger.warning(*args)


def alert(*args) -> None:
    _default_logger.alert(*args)


def debug(*args) -> None:
    _default_logger.debug(*args)
