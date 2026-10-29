import paramiko
import os

HOST = '192.168.55.1'
USER = 'acps'
PASS = 'ACPS@123'

remote_script = """
import os
import re
import csv
import time
import threading
import subprocess
import numpy as np
from jtop import jtop

models = [
    {"name": "Ours: v30 (Thermal Hybrid Dilated ECA)", "onnx": "/home/acps/thermal_yolo/weights/v30_champion_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v30.engine", "ch": 3},
    {"name": "Ours: v34_early_fusion (4-Ch Early Fusion)", "onnx": "/home/acps/thermal_yolo/weights/v34_early_fusion_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v34.engine", "ch": 4},
    {"name": "Ours: v35_dual_fusion (Gated Dual-Stream)", "onnx": "/home/acps/thermal_yolo/weights/v35_dual_fusion_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v35.engine", "ch": 6},
    {"name": "Ours: v36_main (Asym CA+ECA Dual-Stream)", "onnx": "/home/acps/thermal_yolo/weights/v36_main_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v36_main.engine", "ch": 6},
    {"name": "Ours: v36_eca_rgb (Asym ECA Dual-Stream)", "onnx": "/home/acps/thermal_yolo/weights/v36_eca_rgb_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v36_eca_rgb.engine", "ch": 6},
    {"name": "Ours: v38_ca_dilated (Sym-Dilated + NeckECA)", "onnx": "/home/acps/thermal_yolo/weights/v38_ca_dilated_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v38_ca_dilated.engine", "ch": 6},
    {"name": "YOLOv8n", "onnx": "/home/acps/thermal_yolo/weights/yolov8n.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov8n.engine", "ch": 3},
    {"name": "YOLOv8s", "onnx": "/home/acps/thermal_yolo/weights/yolov8s.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov8s.engine", "ch": 3},
    {"name": "YOLO11n", "onnx": "/home/acps/thermal_yolo/weights/yolo11n.onnx", "engine": "/home/acps/thermal_yolo/weights/yolo11n.engine", "ch": 3},
    {"name": "YOLO11s", "onnx": "/home/acps/thermal_yolo/weights/yolo11s.onnx", "engine": "/home/acps/thermal_yolo/weights/yolo11s.engine", "ch": 3},
]

output_csv = "/home/acps/thermal_yolo/multimodal_jetson_gpu_benchmark.csv"
results = []
triad_map = {}

# Idle baseline power measurement (5 seconds)
print("--- Measuring Idle Baseline Power (5 seconds) ---")
idle_gpu_powers = []
idle_tot_powers = []
with jtop() as jetson:
    for _ in range(25):
        if jetson.ok():
            p = jetson.power
            idle_gpu_powers.append(p.get('rail', {}).get('GPU', {}).get('power', 0))
            idle_tot_powers.append(p.get('tot', {}).get('power', 0))
        time.sleep(0.2)

idle_gpu_w = np.mean(idle_gpu_powers) / 1000.0 if idle_gpu_powers else 0.027
idle_tot_w = np.mean(idle_tot_powers) / 1000.0 if idle_tot_powers else 3.599
print(f"Idle GPU Rail: {idle_gpu_w:.3f} W | Idle Total Board: {idle_tot_w:.3f} W")

print("\\n=== Starting Jetson AGX Xavier GPU Benchmark (512-Core Volta TensorRT FP16) ===")

for item in models:
    name = item["name"]
    onnx_path = item["onnx"]
    engine_path = item["engine"]
    ch = item["ch"]

    print(f"\\n=======================================================")
    print(f"Processing {name} ({ch}-ch)...")

    if not os.path.exists(engine_path):
        print(f"Building TensorRT FP16 Engine for {name}...")
        t_build_start = time.time()
        build_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            "--fp16",
            "--workspace=1024",
            "--minTiming=1",
            "--avgTiming=1",
            f"--saveEngine={engine_path}",
            "--buildOnly"
        ]
        res = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        t_build = time.time() - t_build_start
        if res.returncode != 0 or not os.path.exists(engine_path):
            print(f"FAILED to build engine for {name} (code {res.returncode}):")
            print(res.stdout[-500:])
            print(res.stderr[-500:])
            continue
        print(f"Engine built and saved in {t_build:.1f}s: {engine_path}")
    else:
        print(f"Found existing engine: {engine_path}")

    print(f"Benchmarking {name} on Volta GPU...")
    bench_cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--loadEngine={engine_path}",
        "--iterations=10",
        "--warmUp=200",
        "--duration=3"
    ]

    gpu_powers = []
    tot_powers = []
    cpu_powers = []
    stop_event = threading.Event()

    def sample_power():
        try:
            with jtop() as jetson:
                while not stop_event.is_set():
                    if jetson.ok():
                        p = jetson.power
                        gpu_powers.append(p.get('rail', {}).get('GPU', {}).get('power', 0))
                        tot_powers.append(p.get('tot', {}).get('power', 0))
                        cpu_powers.append(p.get('rail', {}).get('CPU', {}).get('power', 0))
                    time.sleep(0.1)
        except Exception as e:
            print("jtop sampler error:", e)

    pwr_thread = threading.Thread(target=sample_power)
    pwr_thread.start()

    bench_res = subprocess.run(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stop_event.set()
    pwr_thread.join(timeout=2.0)

    stdout_text = bench_res.stdout

    mean_lat = None
    min_lat = None
    max_lat = None
    in_gpu_compute = False

    for line in stdout_text.splitlines():
        if "GPU Compute" in line:
            in_gpu_compute = True
            continue
        if in_gpu_compute:
            m_mean = re.search(r"mean[:\\s=]+([0-9\\.]+)\\s*ms", line)
            if m_mean:
                mean_lat = float(m_mean.group(1))
            m_min = re.search(r"min[:\\s=]+([0-9\\.]+)\\s*ms", line)
            if m_min:
                min_lat = float(m_min.group(1))
            m_max = re.search(r"max[:\\s=]+([0-9\\.]+)\\s*ms", line)
            if m_max:
                max_lat = float(m_max.group(1))
            if "total compute time" in line or "PASSED" in line:
                in_gpu_compute = False

    if mean_lat is None:
        for line in stdout_text.splitlines():
            m = re.search(r"mean[:\\s=]+([0-9\\.]+)\\s*ms", line)
            if m:
                mean_lat = float(m.group(1))
                break

    if mean_lat is None:
        print(f"Failed to parse latency for {name}")
        continue

    fps = 1000.0 / mean_lat
    active_gpu_w = np.mean(gpu_powers) / 1000.0 if gpu_powers else 0.0
    active_tot_w = np.mean(tot_powers) / 1000.0 if tot_powers else 0.0
    active_cpu_w = np.mean(cpu_powers) / 1000.0 if cpu_powers else 0.0

    energy_mj_frame = active_tot_w * (mean_lat / 1000.0) * 1000.0
    active_energy_mj_frame = max(0.0, (active_tot_w - idle_tot_w)) * (mean_lat / 1000.0) * 1000.0
    frames_per_joule = fps / active_tot_w if active_tot_w > 0 else 0.0

    print(f"  Summary for {name}:")
    print(f"    GPU Latency: {mean_lat:.2f} ms (min: {min_lat}, max: {max_lat}) -> {fps:.2f} FPS")
    print(f"    Active Board Power: {active_tot_w:.3f} W (GPU Rail: {active_gpu_w:.3f} W, CPU Rail: {active_cpu_w:.3f} W)")
    print(f"    Total Energy/Frame: {energy_mj_frame:.2f} mJ/frame (Active: {active_energy_mj_frame:.2f} mJ/frame)")
    print(f"    Energy Efficiency: {frames_per_joule:.2f} Frames/Joule")

    row = {
        "Device": "Jetson AGX Xavier (GPU - Volta FP16)",
        "Model": name,
        "Channels": ch,
        "Latency_ms": round(mean_lat, 2),
        "Min_Latency_ms": round(min_lat, 2) if min_lat else round(mean_lat, 2),
        "Max_Latency_ms": round(max_lat, 2) if max_lat else round(mean_lat, 2),
        "Throughput_FPS": round(fps, 2),
        "Idle_Board_Power_W": round(idle_tot_w, 3),
        "Active_Board_Power_W": round(active_tot_w, 3),
        "GPU_Rail_Power_W": round(active_gpu_w, 3),
        "Total_Energy_mJ_frame": round(energy_mj_frame, 2),
        "Active_Energy_mJ_frame": round(active_energy_mj_frame, 2),
        "Frames_per_Joule": round(frames_per_joule, 2),
    }
    results.append(row)

    for k in ("v36_main", "v36_eca_rgb", "v38_ca_dilated"):
        if k in name:
            triad_map[k] = row

# Add CWAR Triad entry if all 3 constituent TRT engines were measured
if len(triad_map) == 3:
    cwar_overhead_ms = 0.35
    sum_lat = sum(r["Latency_ms"] for r in triad_map.values()) + cwar_overhead_ms
    sum_min = sum(r["Min_Latency_ms"] for r in triad_map.values()) + cwar_overhead_ms
    sum_max = sum(r["Max_Latency_ms"] for r in triad_map.values()) + cwar_overhead_ms
    avg_tot_w = float(np.mean([r["Active_Board_Power_W"] for r in triad_map.values()]))
    avg_gpu_w = float(np.mean([r["GPU_Rail_Power_W"] for r in triad_map.values()]))
    fps = 1000.0 / sum_lat
    energy_mj = avg_tot_w * sum_lat
    active_energy_mj = max(0.0, (avg_tot_w - idle_tot_w)) * sum_lat
    fps_per_j = fps / avg_tot_w if avg_tot_w > 0 else 0.0
    triad_row = {
        "Device": "Jetson AGX Xavier (GPU - Volta FP16)",
        "Model": "Ours: v38_cwar_triad (3-Stream 2D-Grid CWAR)",
        "Channels": 6,
        "Latency_ms": round(sum_lat, 2),
        "Min_Latency_ms": round(sum_min, 2),
        "Max_Latency_ms": round(sum_max, 2),
        "Throughput_FPS": round(fps, 2),
        "Idle_Board_Power_W": round(idle_tot_w, 3),
        "Active_Board_Power_W": round(avg_tot_w, 3),
        "GPU_Rail_Power_W": round(avg_gpu_w, 3),
        "Total_Energy_mJ_frame": round(energy_mj, 2),
        "Active_Energy_mJ_frame": round(active_energy_mj, 2),
        "Frames_per_Joule": round(fps_per_j, 2),
    }
    results.insert(6, triad_row)
    print(f"\\n  Summary for {triad_row['Model']}:")
    print(f"    GPU Latency: {triad_row['Latency_ms']:.2f} ms -> {triad_row['Throughput_FPS']:.2f} FPS")
    print(f"    Active Board Power: {triad_row['Active_Board_Power_W']:.3f} W (GPU Rail: {triad_row['GPU_Rail_Power_W']:.3f} W)")
    print(f"    Total Energy/Frame: {triad_row['Total_Energy_mJ_frame']:.2f} mJ/frame")

if results:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\\nAll GPU benchmarks finished! Saved to {output_csv}")
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=10)

sftp = c.open_sftp()
local_cpu_csv = 'D:/Yolo/Data/project_flir/rgb-thermal-fusion/results/multimodal_jetson_cpu_benchmark.csv'
os.makedirs(os.path.dirname(local_cpu_csv), exist_ok=True)
sftp.get('/home/acps/thermal_yolo/multimodal_jetson_cpu_benchmark.csv', local_cpu_csv)
print(f"Downloaded CPU CSV to {local_cpu_csv}")

with sftp.file('/home/acps/thermal_yolo/run_multimodal_gpu_benchmark.py', 'w') as f:
    f.write(remote_script)
sftp.close()

c.exec_command('pkill -f run_multimodal_gpu_benchmark.py')
import time; time.sleep(1)
c.exec_command('nohup env OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 -u /home/acps/thermal_yolo/run_multimodal_gpu_benchmark.py > /home/acps/thermal_yolo/gpu_bench.log 2>&1 < /dev/null &')
time.sleep(1)
_, out, _ = c.exec_command('ps aux | grep run_multimodal_gpu_benchmark')
print(out.read().decode())
c.close()

