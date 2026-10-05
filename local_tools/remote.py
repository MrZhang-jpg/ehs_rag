# -*- coding: utf-8 -*-
"""Local helper: run commands / transfer files on the EHS remote host.

Password auth only (session-scoped; nothing is persisted on the remote).

Usage:
    python remote.py run  "<shell command>"
    python remote.py runf <local_script.sh>      # upload script to /tmp and run it
    python remote.py put  <local_path> <remote_path>
    python remote.py get  <remote_path> <local_path>

Password is read from env EHS_SSH_PW.
"""
import os
import sys
import time

import paramiko

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HOST = "connect.westc.seetacloud.com"
PORT = 15117
USER = "root"


def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER,
                password=os.environ["EHS_SSH_PW"],
                timeout=25, banner_timeout=45, auth_timeout=45,
                allow_agent=False, look_for_keys=False)
    return cli


def run(cli: paramiko.SSHClient, cmd: str, timeout: float = 3600) -> int:
    _in, out, err = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    chan = out.channel
    chan.settimeout(1.0)
    while True:
        if chan.recv_ready():
            sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
            sys.stdout.flush()
        elif chan.recv_stderr_ready():
            sys.stderr.write(chan.recv_stderr(65536).decode("utf-8", "replace"))
            sys.stderr.flush()
        elif chan.exit_status_ready():
            # drain whatever is left
            while chan.recv_ready():
                sys.stdout.write(chan.recv(65536).decode("utf-8", "replace"))
            while chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(65536).decode("utf-8", "replace"))
            break
        else:
            time.sleep(0.05)
            continue
        time.sleep(0.01)
    rc = chan.recv_exit_status()
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


def main() -> int:
    mode = sys.argv[1]
    cli = connect()
    try:
        if mode == "run":
            return run(cli, sys.argv[2])
        if mode == "runf":
            local = sys.argv[2]
            remote = "/tmp/_ehs_%d.sh" % int(time.time() * 1000)
            sftp = cli.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            return run(cli, "bash %s" % remote)
        if mode in ("put", "get"):
            sftp = cli.open_sftp()
            if mode == "put":
                sftp.put(sys.argv[2], sys.argv[3])
            else:
                sftp.get(sys.argv[2], sys.argv[3])
            sftp.close()
            print("OK %s %s" % (mode, sys.argv[3]))
            return 0
        raise SystemExit("unknown mode: %s" % mode)
    finally:
        cli.close()


if __name__ == "__main__":
    raise SystemExit(main())
