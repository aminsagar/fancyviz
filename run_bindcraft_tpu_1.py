import os
import subprocess
import queue
import threading
from datetime import datetime

# ===== CONFIG =====
NUM_TPUS = 8
BASE_PORT = 8476
LOG_DIR = "bindcraft_logs"

BINDCRAFT_CMD = [
    "python", "-u",
    "/home/amin_sagar/softwares/BindCraft/bindcraft.py",
    "--settings", "./settings_target/CDCP1_CTD_DefHel_allowCys.json",
    "--filters", "settings_filters/peptide_filters.json",
    "--advanced", "settings_advanced/peptide_3stage_multimer_allowcys.json",
]

# ==================

os.makedirs(LOG_DIR, exist_ok=True)

tpu_queue = queue.Queue()
for i in range(NUM_TPUS):
    tpu_queue.put(i)


def run_bindcraft(job_id: int):
    tpu_id = tpu_queue.get()
    port = BASE_PORT + tpu_id

    env = os.environ.copy()
    env["TPU_VISIBLE_DEVICES"] = str(tpu_id)
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
    env["TPU_WORKER_ID"] = "0"
    env["TPU_NUM_WORKERS"] = "1"
    env["TPU_MESH_CONTROLLER_ADDRESS"] = f"localhost:{port}"
    env["TPU_MESH_CONTROLLER_PORT"] = str(port)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(
        LOG_DIR, f"bindcraft_job{job_id}_tpu{tpu_id}_{timestamp}.log"
    )

    print(f"[Job {job_id}] TPU {tpu_id} → logging to {logfile}")

    with open(logfile, "w") as log:
        try:
            subprocess.run(
                BINDCRAFT_CMD,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        finally:
            print(f"[Job {job_id}] Finished on TPU {tpu_id}")
            tpu_queue.put(tpu_id)


# ===== LAUNCH =====
threads = []
NUM_JOBS = 8

for job_id in range(NUM_JOBS):
    t = threading.Thread(target=run_bindcraft, args=(job_id,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
                                                                                                                           53,5          61%

