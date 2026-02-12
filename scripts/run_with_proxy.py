#!/usr/bin/env python3
"""
Wrapper script that patches socket connections to tunnel through
an HTTP CONNECT proxy for Neo4j Bolt protocol, then launches batch_pipeline.py.

Usage:
    python run_with_proxy.py [batch_pipeline args...]
"""

import os
import socket
import sys

# --- Proxy configuration ---
PROXY_HOST = "httpproxy-us-headless.kubebrain.svc.pjlab.local"
PROXY_PORT = 3128
# Hosts that need tunnelling (Neo4j Bolt uses raw TCP, not HTTP)
TUNNEL_HOSTS = {"115.190.6.177"}

_original_connect = socket.socket.connect


def _patched_connect(self, address):
    """Intercept socket.connect to tunnel through HTTP CONNECT proxy."""
    try:
        host = address[0]
        port = address[1]
    except (IndexError, TypeError):
        return _original_connect(self, address)

    if host in TUNNEL_HOSTS:
        # First connect to the proxy
        _original_connect(self, (PROXY_HOST, PROXY_PORT))
        # Send HTTP CONNECT request
        connect_req = (
            f"CONNECT {host}:{port} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"\r\n"
        )
        self.sendall(connect_req.encode())
        # Read proxy response
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.recv(4096)
            if not chunk:
                raise ConnectionError(
                    f"Proxy CONNECT to {host}:{port} closed prematurely"
                )
            buf += chunk
        status_line = buf.split(b"\r\n")[0].decode()
        if "200" not in status_line:
            raise ConnectionError(f"Proxy CONNECT failed: {status_line}")
        # Tunnel established — the socket is now connected to the target
        return
    else:
        return _original_connect(self, address)


def main():
    # Apply monkey-patches
    socket.socket.connect = _patched_connect

    # Also patch create_connection for any code that uses it
    _original_create_connection = socket.create_connection

    def _patched_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                                   source_address=None, **kwargs):
        host, port = address
        if host in TUNNEL_HOSTS:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                s.settimeout(timeout)
            # _patched_connect will handle the tunnelling
            s.connect((host, port))
            return s
        return _original_create_connection(address, timeout=timeout,
                                           source_address=source_address, **kwargs)

    socket.create_connection = _patched_create_connection

    # Set HTTP proxy env vars for LLM API calls (requests/urllib3/httpx)
    os.environ.setdefault("http_proxy",
                          f"http://{PROXY_HOST}:{PROXY_PORT}")
    os.environ.setdefault("https_proxy",
                          f"http://{PROXY_HOST}:{PROXY_PORT}")
    os.environ.setdefault("HTTP_PROXY",
                          f"http://{PROXY_HOST}:{PROXY_PORT}")
    os.environ.setdefault("HTTPS_PROXY",
                          f"http://{PROXY_HOST}:{PROXY_PORT}")
    os.environ.setdefault("no_proxy",
                          "10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn")

    # Switch to the backend directory and run batch_pipeline
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "crossbar_llm", "backend")
    backend_dir = os.path.abspath(backend_dir)
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)

    # Parse --config from args for later use
    config_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ("--config", "-c") and i + 2 <= len(sys.argv[1:]):
            config_path = sys.argv[1:][i + 1]
            break

    # Step 1: Run batch pipeline
    sys.argv = ["batch_pipeline.py"] + sys.argv[1:]
    from batch_pipeline import main as bp_main
    bp_main()

    # Step 2: Run compare_results on latest run
    from pathlib import Path
    project_root = Path(backend_dir).parent.parent
    batch_output_dir = project_root / "batch_output"
    from compare_results import find_latest_run
    latest_run = find_latest_run(batch_output_dir)
    if latest_run:
        print(f"\n{'='*46}")
        print("  Generating Comparison Report")
        print(f"{'='*46}\n")
        sys.argv = ["compare_results.py", "--run-dir", str(latest_run), "--format", "both"]
        from compare_results import main as cr_main
        cr_main()

        # Step 3: Run evaluate_results (judge + md reports)
        print(f"\n{'='*46}")
        print("  Running LLM Judge & Generating Reports")
        print(f"{'='*46}\n")
        eval_args = ["evaluate_results.py", "--run-dir", str(latest_run)]
        if config_path:
            eval_args += ["--config", config_path]
        sys.argv = eval_args
        from evaluate_results import main as er_main
        er_main()
    else:
        print("No run directory found, skipping comparison and evaluation.")


if __name__ == "__main__":
    main()
