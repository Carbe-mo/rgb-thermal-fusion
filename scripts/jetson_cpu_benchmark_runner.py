import paramiko
import os

HOST = '192.168.55.1'
USER = 'acps'
PASS = 'ACPS@123'

remote_script = """
import time
import os
import csv
import numpy as np
import onnxruntime as ort
from jtop import jtop

models = [
    {"name": "Ours: v30 (Thermal Hybrid Dilated ECA)", "file": "/home/acps/thermal_yolo/weights/v30_champion_best.onnx", "channels": 3},
    {"name": "Ours: v34_early_fusion (4-Ch Early Fusion)", "file": "/home/acps/thermal_yolo/weights/v34_early_fusion_best.onnx", "channels": 4},
    {"name": "Ours: v35_dual_fusion (Gated Dual-Stream)", "file": "/home/acps/thermal_yolo/weights/v35_dual_fusion_best.onnx", "channels": 6},
    {"name": "Ours: v36_main (Asym CA+ECA Dual-Stream)", "file": "/home/acps/thermal_yolo/weights/v36_main_best.onnx", "channels": 6},
    {"name": "Ours: v36_eca_rgb (Asym ECA Dual-Stream)", "file": "/home/acps/thermal_yolo/weights/v36_eca_rgb_best.onnx", "channels": 6},
    {"name": "Ours: v38_ca_dilated (Sym-Dilated + NeckECA)", "file": "/home/acps/thermal_yolo/weights/v38_ca_dilated_best.onnx", "channels": 6},
    {"name": "Ours: v38_cwar_triad (3-Stream 2D-Grid CWAR)", "file": "CWAR_TRIAD", "channels": 6},
    {"name": "YOLOv8n", "file": "/home/acps/thermal_yolo/weights/yolov8n.onnx", "channels": 3},
    {"name": "YOLOv8s", "file": "/home/acps/thermal_yolo/weights/yolov8s.onnx", "channels": 3},
    {"name": "YOLOv10n", "file": "/home/acps/thermal_yolo/weights/yolov10n.onnx", "channels": 3},
    {"name": "YOLOv10s", "file": "/home/acps/thermal_yolo/weights/yolov10s.onnx", "channels": 3},
    {"name": "YOLO11n", "file": "/home/acps/thermal_yolo/weights/yolo11n.onnx", "channels": 3},
    {"name": "YOLO11s", "file": "/home/acps/thermal_yolo/weights/yolo11s.onnx", "channels": 3},
]

results = []
output_csv = "/home/acps/thermal_yolo/multimodal_jetson_cpu_benchmark.csv"

# Idle baseline power measurement (5 seconds)
print("--- Measuring Idle Baseline Power (5 seconds) ---")
idle_cpu_powers = []
idle_tot_powers = []
with jtop() as jetson:
    for _ in range(25):
        if jetson.ok():
            p = jetson.power
            idle_cpu_powers.append(p.get('rail', {}).get('CPU', {}).get('power', 0))
            idle_tot_powers.append(p.get('tot', {}).get('power', 0))
        time.sleep(0.2)

idle_cpu_w = np.mean(idle_cpu_powers) / 1000.0 if idle_cpu_powers else 2.5
idle_tot_w = np.mean(idle_tot_powers) / 1000.0 if idle_tot_powers else 3.8
print(f"Idle CPU Rail: {idle_cpu_w:.3f} W | Idle Total Board: {idle_tot_w:.3f} W")

print("\\n=== Starting Jetson AGX Xavier CPU Benchmark (8 Carmel ARMv8.2 Cores) ===")
with jtop() as jetson:
    for item in models:
        name = item["name"]
        path = item["file"]
        ch = item["channels"]
        print(f"\\n--- Benchmarking {name} on CPU ---")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 8
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        dummy_input = np.random.randn(1, ch, 640, 640).astype(np.float32)

        if path == "CWAR_TRIAD":
            p1 = "/home/acps/thermal_yolo/weights/v36_main_best.onnx"
            p2 = "/home/acps/thermal_yolo/weights/v36_eca_rgb_best.onnx"
            p3 = "/home/acps/thermal_yolo/weights/v38_ca_dilated_best.onnx"
            s1 = ort.InferenceSession(p1, sess_options, providers=['CPUExecutionProvider'])
            s2 = ort.InferenceSession(p2, sess_options, providers=['CPUExecutionProvider'])
            s3 = ort.InferenceSession(p3, sess_options, providers=['CPUExecutionProvider'])
            in1 = s1.get_inputs()[0].name
            in2 = s2.get_inputs()[0].name
            in3 = s3.get_inputs()[0].name
            w_cls = np.array([0.42, 0.38, 0.20], dtype=np.float32).reshape(3, 1, 1, 1)
            w_box = np.array([0.42, 0.38, 0.20], dtype=np.float32).reshape(3, 1, 1, 1)

            def run_once():
                y1 = s1.run(None, {in1: dummy_input})[0]
                y2 = s2.run(None, {in2: dummy_input})[0]
                y3 = s3.run(None, {in3: dummy_input})[0]
                ys = np.stack([y1, y2, y3], axis=0)
                boxes = ys[:, :, :4, :]
                probs = ys[:, :, 4:, :]
                fused_probs = (probs * w_cls).sum(axis=0)
                conf_peak = np.maximum(probs.max(axis=2, keepdims=True), 1e-6)
                psi = w_box * (conf_peak ** 1.25)
                psi_norm = psi / np.maximum(psi.sum(axis=0, keepdims=True), 1e-6)
                fused_boxes = (boxes * psi_norm).sum(axis=0)
                return np.concatenate([fused_boxes, fused_probs], axis=1)
        else:
            if not os.path.exists(path):
                print(f"ERROR: File not found {path}")
                continue
            try:
                session = ort.InferenceSession(path, sess_options, providers=['CPUExecutionProvider'])
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
            input_name = session.get_inputs()[0].name

            def run_once():
                return session.run(None, {input_name: dummy_input})

        # Warmup (2 iterations)
        for _ in range(2):
            run_once()

        latencies = []
        cpu_rail_powers = []
        tot_board_powers = []

        # 10 timed test runs
        for i in range(10):
            t0 = time.perf_counter()
            run_once()
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            cpu_p, tot_p = 0, 0
            if jetson.ok():
                p = jetson.power
                cpu_p = p.get('rail', {}).get('CPU', {}).get('power', 0)
                tot_p = p.get('tot', {}).get('power', 0)
                cpu_rail_powers.append(cpu_p)
                tot_board_powers.append(tot_p)

            print(f"  [{name}] Run {i+1}/10: {dt:.2f} ms | CPU rail: {cpu_p} mW | Total: {tot_p} mW")

        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        fps = 1000.0 / mean_lat
        mean_cpu_w = np.mean(cpu_rail_powers) / 1000.0 if cpu_rail_powers else 0.0
        mean_tot_w = np.mean(tot_board_powers) / 1000.0 if tot_board_powers else 0.0

        # Energy calculations
        energy_mj_frame = mean_tot_w * (mean_lat / 1000.0) * 1000.0
        active_energy_mj_frame = max(0.0, (mean_tot_w - idle_tot_w)) * (mean_lat / 1000.0) * 1000.0
        frames_per_joule = fps / mean_tot_w if mean_tot_w > 0 else 0.0

        print(f"  Summary for {name}:")
        print(f"    Latency: {mean_lat:.2f} +/- {std_lat:.2f} ms ({fps:.2f} FPS)")
        print(f"    Total Board Power: {mean_tot_w:.3f} W (CPU rail: {mean_cpu_w:.3f} W)")
        print(f"    Energy/Frame: {energy_mj_frame:.1f} mJ/frame (Active: {active_energy_mj_frame:.1f} mJ/frame)")
        print(f"    Efficiency: {frames_per_joule:.2f} Frames/Joule")

        results.append({
            "Device": "Jetson AGX Xavier (CPU - 8x Carmel ARMv8.2)",
            "Model": name,
            "Channels": ch,
            "Latency_ms": round(mean_lat, 2),
            "Latency_std_ms": round(std_lat, 2),
            "Throughput_FPS": round(fps, 2),
            "Idle_Board_Power_W": round(idle_tot_w, 3),
            "Active_Board_Power_W": round(mean_tot_w, 3),
            "CPU_Rail_Power_W": round(mean_cpu_w, 3),
            "Total_Energy_mJ_frame": round(energy_mj_frame, 1),
            "Active_Energy_mJ_frame": round(active_energy_mj_frame, 1),
            "Frames_per_Joule": round(frames_per_joule, 2),
        })

with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\\nAll CPU benchmarks completed! Results saved to {output_csv}")
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

# Upload script
sftp = c.open_sftp()
with sftp.file('/home/acps/thermal_yolo/run_multimodal_cpu_benchmark.py', 'w') as f:
    f.write(remote_script)
sftp.close()

# Execute script with unbuffered output
stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 -u /home/acps/thermal_yolo/run_multimodal_cpu_benchmark.py')
for line in stdout:
    print(line, end='', flush=True)

err = stderr.read().decode()
if err:
    print("STDERR:", err)

# Download CSV
sftp = c.open_sftp()
local_csv = 'D:/Yolo/Data/project_flir/rgb-thermal-fusion/results/multimodal_jetson_cpu_benchmark.csv'
os.makedirs(os.path.dirname(local_csv), exist_ok=True)
sftp.get('/home/acps/thermal_yolo/multimodal_jetson_cpu_benchmark.csv', local_csv)
sftp.close()
c.close()

print(f"CSV downloaded to {local_csv}")
