"""The report modes, one module per family, each mode returning a `Report` that the CLI
prints and the MCP server returns as text."""
from typing import NamedTuple


class Report(NamedTuple):
    """What a report mode found: the lines to show, and how many findings they hold."""
    lines: list
    count: int = 0

    @property
    def text(self):
        return "\n".join(self.lines)

    def show(self):
        """Print the lines as the CLI shows them; nothing at all for an empty report."""
        if self.lines:
            print(self.text)
