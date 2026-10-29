"""
scripts/automated_power_suite.py
================================
Fully Automated End-to-End Joulescope JS220 + SSH Benchmark for Edge AI.
Directly triggers exactly 10 inferences on the edge board (Rock Pi / Raspberry Pi)
over SSH while capturing high-speed microjoule-accurate energy data from
the connected Joulescope JS220.
"""

import csv
import re
import sys
import time
from pathlib import Path
import numpy as np
import paramiko
import joulescope

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "results" / "power_benchmark_results.csv"

# Models to evaluate (Name, weights_path, channels)
DEFAULT_MODELS = [
    ("Ours: v36_main (Asym CA+ECA)", "weights/v36_main_best.pt", 6),
    ("Ours: v36_eca_rgb (Asym ECA)", "weights/v36_eca_rgb_best.pt", 6),
    ("Ours: v38_ca_dilated (Sym-Dil+NeckECA)", "weights/v38_ca_dilated_best.pt", 6),
    ("Ours: v38_cwar_triad (3-Stream CWAR)", "CWAR_TRIAD", 6),
    ("Ours: v30 (Thermal Dilated ECA)", "weights/v30_champion_best.pt", 3),
    ("YOLOv8n", "weights/yolov8n.pt", 3),
    ("YOLOv8s", "weights/yolov8s.pt", 3),
    ("YOLOv10n", "weights/yolov10n.pt", 3),
    ("YOLOv10s", "weights/yolov10s.pt", 3),
    ("YOLOv11n", "weights/yolo11n.pt", 3),
    ("YOLOv11s", "weights/yolo11s.pt", 3),
]


def save_power_row(row_dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()
    headers = list(row_dict.keys())

    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"  --> Saved record to {OUTPUT_CSV}")


def run_automated_power_benchmark(target_ip="192.168.137.168", username="rock", password="rock",
                                  runs=10, models=None, remote_dir="~/rgb-thermal-fusion",
                                  device_name="Rock Pi 4C+"):
    if models is None:
        models = DEFAULT_MODELS

    print("=" * 80)
    print(f"  AUTOMATED JOULESCOPE JS220 POWER BENCHMARK: {device_name} ({username}@{target_ip})")
    print(f"  Inferences per model: {runs}")
    print(f"  Total models to test: {len(models)}")
    print("=" * 80)

    # 1. Connect Joulescope
    print("\n[1/4] Connecting to Joulescope JS220...")
    device = joulescope.scan_require_one(config="auto")
    device.open()
    print(f"  Connected: {device}")

    # 2. Connect SSH
    print("\n[2/4] Connecting SSH to Edge Device...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(target_ip, username=username, password=password, timeout=15)
    print(f"  SSH Connected to {username}@{target_ip}!")

    results = []

    try:
        # 3. Measure Idle Baseline
        print("\n[3/4] Measuring 15-second Idle Baseline...")
        time.sleep(2)
        data_idle = device.read(duration=15.0)
        v_idle = np.abs(data_idle[:, 1])
        i_idle = np.abs(data_idle[:, 0])
        p_idle = float(np.mean(v_idle * i_idle))
        mean_v_idle = float(np.mean(v_idle))
        mean_i_idle_ma = float(np.mean(i_idle)) * 1000.0
        print(f"  -> Stable Idle Baseline: {p_idle:.3f} W (Voltage: {mean_v_idle:.2f} V, Current: {mean_i_idle_ma:.1f} mA)")

        # 4. Measure Models
        print("\n[4/4] Starting Synchronized Multi-Model Inferences...")
        for name, weights, ch in models:
            print("\n" + "=" * 80)
            print(f"  PROFILING: {name} (Weights: {weights}, Channels: {ch})")
            print("=" * 80)

            # Start remote runner
            cmd = f"cd {remote_dir} && python3 -u scripts/measure_edge_power.py --weights {weights} --runs {runs} --channels {ch}"
            print(f"  [Remote] Launching: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)

            # Wait for board to finish loading and warm-up
            print("  [Handshake] Waiting for model loading and warm-up on edge board...")
            ready = False
            while True:
                line = stdout.readline()
                if not line:
                    break
                stripped = line.strip()
                print(f"    [Board] {stripped}")
                if "[SYNC_READY]" in stripped:
                    ready = True
                    break

            if not ready:
                err_msg = stderr.read().decode()
                print(f"  [ERROR] Board failed to signal READY:\n{err_msg}")
                continue

            # Start high-resolution Joulescope recording
            device.parameter_set("buffer_duration", 180)
            t0 = time.time()
            device.start()
            print(f"  🔴 [Joulescope] High-speed recording ACTIVE! Triggering {runs} inferences...")

            # Send trigger to start exact N inferences
            stdin.write("START\n")
            stdin.flush()

            # Wait for inferences to complete
            done_line = None
            while True:
                line = stdout.readline()
                if not line:
                    break
                stripped = line.strip()
                print(f"    [Board] {stripped}")
                if "[SYNC_DONE" in stripped:
                    done_line = stripped
                    break

            # Stop Joulescope immediately
            device.stop()
            t_active = time.time() - t0
            print(f"  ⏹️ [Joulescope] Stopped. Elapsed window: {t_active:.3f}s")

            if not done_line:
                err_msg = stderr.read().decode()
                print(f"  [ERROR] Board did not report completion:\n{err_msg}")
                continue

            # Parse stats from board output: [SYNC_DONE duration_s=4.0309 avg_ms=2015.36 fps=0.50]
            m_dur = re.search(r"duration_s=([0-9\.]+)", done_line)
            m_ms = re.search(r"avg_ms=([0-9\.]+)", done_line)
            m_fps = re.search(r"fps=([0-9\.]+)", done_line)

            inf_duration_s = float(m_dur.group(1)) if m_dur else t_active
            avg_latency_ms = float(m_ms.group(1)) if m_ms else (inf_duration_s / runs * 1000.0)
            fps = float(m_fps.group(1)) if m_fps else (1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0)

            # Retrieve Joulescope recorded samples
            start_id, end_id = device.stream_buffer.sample_id_range
            samples = device.stream_buffer.samples_get(start_id, end_id, fields=["current", "voltage"])
            i_raw = np.abs(samples["signals"]["current"]["value"])
            v_raw = np.abs(samples["signals"]["voltage"]["value"])

            # Compute power and energy metrics
            power_w = i_raw * v_raw
            p_active = float(np.mean(power_w))
            total_energy_j = p_active * inf_duration_s
            dynamic_power_w = max(0.0, p_active - p_idle)

            energy_per_frame_j = total_energy_j / runs
            energy_per_frame_mj = energy_per_frame_j * 1000.0
            frames_per_joule = (1.0 / energy_per_frame_j) if energy_per_frame_j > 0 else 0.0

            mean_volts = float(np.mean(v_raw))
            mean_curr_ma = float(np.mean(i_raw)) * 1000.0

            print("\n  " + "-" * 55)
            print(f"  MEASUREMENT RESULTS: {name}")
            print("  " + "-" * 55)
            print(f"  -> Inferences Completed: {runs} frames in {inf_duration_s:.3f} s")
            print(f"  -> Mean Latency        : {avg_latency_ms:.2f} ms ({fps:.2f} FPS)")
            print(f"  -> Mean Voltage        : {mean_volts:.2f} V")
            print(f"  -> Mean Current        : {mean_curr_ma:.1f} mA")
            print(f"  -> Idle Baseline Power : {p_idle:.3f} W")
            print(f"  -> Active Power        : {p_active:.3f} W")
            print(f"  -> Dynamic Power Delta : {dynamic_power_w:.3f} W")
            print(f"  -> Total Energy Spent  : {total_energy_j:.3f} Joules")
            print(f"  -> Energy per Frame    : {energy_per_frame_mj:.2f} mJ / frame ⚡")
            print(f"  -> Energy Efficiency   : {frames_per_joule:.2f} Frames / Joule")
            print("  " + "-" * 55)

            rec = {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Target_Device": device_name,
                "Model_Name": name,
                "Weights": weights,
                "Inferences": runs,
                "Total_Duration_s": round(inf_duration_s, 3),
                "Avg_Latency_ms": round(avg_latency_ms, 2),
                "Avg_FPS": round(fps, 2),
                "Idle_Power_W": round(p_idle, 3),
                "Active_Power_W": round(p_active, 3),
                "Dynamic_Power_W": round(dynamic_power_w, 3),
                "Total_Energy_J": round(total_energy_j, 3),
                "Energy_per_Frame_mJ": round(energy_per_frame_mj, 2),
                "Frames_per_Joule": round(frames_per_joule, 2),
                "Mean_Voltage_V": round(mean_volts, 2),
                "Mean_Current_mA": round(mean_curr_ma, 1),
            }
            save_power_row(rec)
            results.append(rec)

            # Let device cool down slightly before next model
            print("  [Cool-down] Pausing 5s for thermal stabilization...")
            time.sleep(5)

        # Print Final Summary Table (IEEE Sensors Journal Table VII format)
        print("\n" + "=" * 90)
        print(f"{'MODEL':<25} | {'LATENCY (ms)':<14} | {'FPS':<6} | {'POWER (W)':<10} | {'ENERGY (mJ/frame)':<18} | {'EFF (fps/W)':<10}")
        print("-" * 90)
        for r in results:
            eff_fps_w = r['Avg_FPS'] / r['Active_Power_W'] if r['Active_Power_W'] > 0 else 0
            print(f"{r['Model_Name']:<25} | {r['Avg_Latency_ms']:<14.1f} | {r['Avg_FPS']:<6.2f} | {r['Active_Power_W']:<10.3f} | {r['Energy_per_Frame_mJ']:<18.1f} | {eff_fps_w:<10.2f}")
        print("=" * 90)

    finally:
        ssh.close()
        device.close()
        print("\nJoulescope device and SSH connection closed cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Joulescope JS220 + SSH Edge Power & Latency Suite")
    parser.add_argument("--ip", type=str, default="192.168.137.168", help="Target edge device IP address")
    parser.add_argument("--user", type=str, default="rock", help="SSH username (e.g., rock, pi)")
    parser.add_argument("--password", type=str, default="rock", help="SSH password")
    parser.add_argument("--remote-dir", type=str, default="~/rgb-thermal-fusion", help="Path to cloned repo on edge device")
    parser.add_argument("--device-name", type=str, default="Rock Pi 4C+", help="Device name for CSV logging")
    parser.add_argument("--runs", type=int, default=10, help="Number of synchronized inferences per model")
    parser.add_argument("--only-multimodal", action="store_true", help="Only run multimodal models + v30 champion")
    args = parser.parse_args()

    selected_models = DEFAULT_MODELS[:5] if args.only_multimodal else DEFAULT_MODELS
    run_automated_power_benchmark(
        target_ip=args.ip,
        username=args.user,
        password=args.password,
        runs=args.runs,
        models=selected_models,
        remote_dir=args.remote_dir,
        device_name=args.device_name,
    )
