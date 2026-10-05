# -*- coding: utf-8 -*-
"""Local SSH port-forward: 127.0.0.1:<local_port> -> remote 127.0.0.1:<remote_port>.

Password auth (EHS_SSH_PW env). Runs until killed.

Usage: python tunnel.py [local_port] [remote_port]   (default 8000 8000)
"""
import select
import socket
import sys
import threading

import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 15117
USER = "root"


def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER,
                password=__import__("os").environ["EHS_SSH_PW"],
                timeout=25, banner_timeout=45, auth_timeout=45,
                allow_agent=False, look_for_keys=False)
    return cli


def bridge(chan, conn):
    try:
        while True:
            r, _w, _x = select.select([conn, chan], [], [], 60)
            if conn in r:
                data = conn.recv(8192)
                if not data:
                    break
                chan.sendall(data)
            if chan in r:
                data = chan.recv(8192)
                if not data:
                    break
                conn.sendall(data)
    except Exception:
        pass
    finally:
        try:
            chan.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def main() -> int:
    local_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    remote_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    cli = connect()
    transport = cli.get_transport()
    transport.set_keepalive(30)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(128)
    print("TUNNEL_READY 127.0.0.1:%d -> remote 127.0.0.1:%d" % (local_port, remote_port),
          flush=True)

    while True:
        conn, addr = server.accept()
        try:
            chan = transport.open_channel(
                "direct-tcpip", ("127.0.0.1", remote_port), addr)
        except Exception as e:
            print("open_channel failed:", e, flush=True)
            conn.close()
            continue
        threading.Thread(target=bridge, args=(chan, conn), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
