# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rich terminal dashboard for the sandbox server."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if TYPE_CHECKING:
    from .state_store import AuditEntry, SandboxState


class SandboxDashboard:
    """Live terminal dashboard showing tool calls and blast radius."""

    def __init__(self, state: "SandboxState", port: int = 8199) -> None:
        self._state = state
        self._port = port
        self._console = Console() if HAS_RICH else None
        self._live: "Live | None" = None
        self._recent_calls: list["AuditEntry"] = []
        self._max_recent = 15

    def start(self) -> None:
        if not HAS_RICH:
            print("[sandbox] rich not installed — dashboard disabled, using plain logging")
            return
        self._live = Live(self._render(), console=self._console, refresh_per_second=2)
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def on_tool_call(self, entry: "AuditEntry") -> None:
        self._recent_calls.append(entry)
        if len(self._recent_calls) > self._max_recent:
            self._recent_calls = self._recent_calls[-self._max_recent:]

        if self._live:
            self._live.update(self._render())
        elif HAS_RICH:
            self._print_entry(entry)
        else:
            self._print_plain(entry)

    def _print_plain(self, entry: "AuditEntry") -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
        marker = "!!" if entry.violations else "ok"
        print(f"[{ts}] [{marker}] {entry.tool_name} -> {entry.side_effect}", flush=True)
        for v in entry.violations:
            print(f"         VIOLATION: {v}", flush=True)

    def _print_entry(self, entry: "AuditEntry") -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
        if entry.violations:
            self._console.print(f"  [{ts}] [bold red]⚠[/] {entry.tool_name}  [red]{entry.violations}[/]")
        else:
            self._console.print(f"  [{ts}] [green]✓[/] {entry.tool_name}")

    def _render(self) -> Panel:
        layout = Table.grid(padding=(0, 2))
        layout.add_column()

        # Tool calls section
        calls_table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
        calls_table.add_column("time", style="dim", width=10)
        calls_table.add_column("status", width=3)
        calls_table.add_column("tool", width=35)
        calls_table.add_column("detail")

        for entry in self._recent_calls[-10:]:
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            if entry.violations:
                status = Text("⚠", style="bold red")
                detail = Text(", ".join(entry.violations), style="red")
            else:
                status = Text("✓", style="green")
                detail = Text(entry.side_effect, style="dim")
            calls_table.add_row(ts, status, entry.tool_name, detail)

        # Blast radius section
        br = self._state.get_blast_radius()
        blast_lines = []
        for tag in sorted(set(t for e in self._recent_calls for t in e.blast_radius_tags)):
            blast_lines.append(Text(f"  \U0001f534 {tag} (critical)", style="bold red"))

        # State summary
        summary = self._state.get_state_summary()
        state_parts = []
        for domain, counts in summary.items():
            total = sum(counts.values())
            if total > 0:
                state_parts.append(f"{domain}: {total}")

        # Assemble
        content = Table.grid()
        content.add_column()
        content.add_row(Text("TOOL CALLS", style="bold underline"))
        content.add_row(calls_table)
        content.add_row(Text(""))

        if blast_lines:
            content.add_row(Text("BLAST RADIUS", style="bold underline"))
            for line in blast_lines:
                content.add_row(line)
            content.add_row(Text(""))

        state_text = " | ".join(state_parts) if state_parts else "no operations yet"
        content.add_row(Text(f"STATE: {state_text}", style="dim"))

        return Panel(
            content,
            title=f"ErosionBench Sandbox ── localhost:{self._port}",
            border_style="cyan",
        )
