"""
Launcher script for Automated AI Video Dubbing Studio Web Application.
Starts the FastAPI server with Uvicorn and automatically launches the user's browser.
"""

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn
from rich.console import Console
from rich.panel import Panel

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()


def open_browser_delayed(url: str, delay: float = 1.2):
    """Open the browser after a brief pause so Uvicorn has bound to the port."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception as e:
        console.print(f"[yellow]Could not automatically open browser: {e}[/yellow]")


def find_available_port(host: str, preferred_port: int, max_attempts: int = 20) -> int:
    """Find an available port starting from preferred_port."""
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return preferred_port


def main():
    parser = argparse.ArgumentParser(description="Launch the Dubber Studio Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    active_port = find_available_port(args.host, args.port)
    if active_port != args.port:
        console.print(
            f"[bold yellow]Notice: Port {args.port} is already in use. Automatically routed to port {active_port}.[/bold yellow]"
        )

    url = f"http://{args.host}:{active_port}"

    console.print(
        Panel.fit(
            f"[bold cyan]AI Video Dubbing Studio[/bold cyan]\n"
            f"[green]Web Interface:[/green] [bold underline white]{url}[/bold underline white]\n\n"
            f"[dim]* Automatic vocal pitch F0 gender classification (Male / Female)\n"
            f"* Real-time SSE pipeline streaming (5 stages)\n"
            f"* Interactive HTML5 video player with English subtitle support[/dim]",
            border_style="cyan",
            title="[bold yellow]Dubber Studio Server[/bold yellow]",
        )
    )

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    try:
        uvicorn.run(
            "server:app",
            host=args.host,
            port=active_port,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dubber Studio server stopped.[/yellow]")
        sys.exit(0)
    except OSError as e:
        console.print(f"\n[bold red]Network Socket Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
