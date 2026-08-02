import socket
import sys
import time

ports = [int(p) for p in sys.argv[1:]]


def up(p):
    try:
        s = socket.create_connection(("127.0.0.1", p), 1)
        s.close()
        return True
    except Exception:
        return False


deadline = time.time() + 180
while time.time() < deadline:
    if all(up(p) for p in ports):
        print("ready:", ports)
        sys.exit(0)
    time.sleep(2)
print("timeout waiting for", ports)
sys.exit(1)
