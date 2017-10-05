#!/usr/bin/env python
"""Script for launching a remote notebook server
"""
from subprocess import Popen, run, PIPE
import signal
import sys
import re
import os
import time

os.environ['JUPYTER_RUNTIME_DIR'] = os.path.join(os.environ['HOME'], ".local/runtime")


# handle Cntrl-C from user
def signal_handler(signal, frame):
    ch = input("Kill Server and portforwarding processes?")
    if ch.lower().startswith("y"):
        print("Killing server")
        notebook_proc.kill()
        print("Killing ssh processes")
        for pf in port_forwards:
            pf.kill()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

login_port ="10001"
compute_port = "8888"
login_nodes = ["log-0", "log-1"]


print("Opening ssh port forwards")
port_forwards = [Popen(["ssh", "-N",
                        "-R", f"{login_port}:127.0.0.1:{compute_port}",
                        host])
                 for host in login_nodes]




print("Starting notebook server")
notebook_proc = Popen(["jupyter", "notebook",
                     "--no-browser",
                       "--port", compute_port],
                       stderr=PIPE)

for line in iter(notebook_proc.stderr):
    print(re.sub(str(compute_port), str(login_port), 
        line.decode("utf-8").strip("\n")))
notebook_proc.wait()
