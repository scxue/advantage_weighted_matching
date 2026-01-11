import os
NUM_DEVICES = 8
USED_DEVICES = set()

# reward='deqa'
reward='geneval'
if reward=='deqa':
    port=18086
if reward=='geneval':
    port=18085

def pre_fork(server, worker):
    # runs on server
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)

def post_fork(server, worker):
    # runs on worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)

def child_exit(server, worker):
    # runs on server
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)

# Gunicorn Configuration
bind = f"127.0.0.1:{port}"
# for cross node access
# bind = "0.0.0.0:18085"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120
