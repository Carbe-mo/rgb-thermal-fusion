"""
scripts/joulescope_capture.py
==============================
Precision Joulescope JS220 / JS110 Energy & Power Capture Tool for Edge AI.

Features:
  - Exact-inference interactive capture: Press [ENTER] to start recording when
    your board starts the 10 inferences, and press [ENTER] when it finishes!
  - Calculates true Total Energy (Joules), Active Power (Watts), and
    Energy per Inference (mJ/frame).
  - Multi-model suite mode (--suite) to benchmark all models sequentially
    for IEEE Sensors Journal Table VII.

Usage (Host PC):
  # Single model test (10 inferences):
  python scripts/joulescope_capture.py --device "Rock Pi 4C+" --model "Dilated_ECA_v30" --inferences 10

  # Full suite mode (walks through all 8 models one by one):
  python scripts/joulescope_capture.py --device "Rock Pi 4C+" --suite --inferences 10
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "results" / "power_benchmark_results.csv"

ALL_MODELS = [
    ("Ours: v36_main (Asym CA+ECA)", "weights/v36_main_best.pt"),
    ("Ours: v36_eca_rgb (Asym ECA)", "weights/v36_eca_rgb_best.pt"),
    ("Ours: v38_ca_dilated (Sym-Dil+NeckECA)", "weights/v38_ca_dilated_best.pt"),
    ("Ours: v38_cwar_triad (3-Stream CWAR)", "CWAR_TRIAD"),
    ("Ours: v30 (Thermal Dilated ECA)", "weights/v30_champion_best.pt"),
    ("YOLOv8n", "weights/yolov8n.pt"),
    ("YOLOv8s", "weights/yolov8s.pt"),
    ("YOLOv10n", "weights/yolov10n.pt"),
    ("YOLOv10s", "weights/yolov10s.pt"),
    ("YOLOv11n", "weights/yolo11n.pt"),
    ("YOLOv11s", "weights/yolo11s.pt"),
]


def check_joulescope():
    try:
        import joulescope
        return joulescope
    except ImportError:
        print("\n[ERROR] 'joulescope' package not found.")
        print("Please install via: pip install joulescope\n")
        sys.exit(1)


def capture_idle(device, duration_sec: float = 15.0):
    print(f"\n[Joulescope] Measuring Idle Baseline for {duration_sec:.1f}s...")
    print("  Ensure edge device is powered ON and completely idle (no inference running).")
    input("  Press [ENTER] to record Idle baseline...")
    
    data = device.read(duration=duration_sec)
    if isinstance(data, dict):
        i = np.asarray(data["signals"]["current"]["value"])
        v = np.asarray(data["signals"]["voltage"]["value"])
    else:
        i = data[:, 0]
        v = data[:, 1]
        
    power_w = i * v
    mean_p = float(np.mean(power_w))
    mean_v = float(np.mean(v))
    mean_i_ma = float(np.mean(i)) * 1000.0
    
    print(f"  -> Idle Baseline: {mean_p:.3f} W ({mean_v:.2f} V, {mean_i_ma:.1f} mA)")
    return {"idle_power_w": mean_p, "voltage_v": mean_v, "current_ma": mean_i_ma}


def capture_active_interactive(device, model_name: str, num_inferences: int = 10, buffer_dur: int = 180):
    try:
        device.parameter_set("buffer_duration", buffer_dur)
    except Exception:
        pass

    print("\n" + "-" * 70)
    print(f" ACTIVE INFERENCE CAPTURE: {model_name} ({num_inferences} Inferences)")
    print("-" * 70)
    print(" 1. Have your edge board command ready to execute, e.g.:")
    print(f"    python scripts/benchmark_edge.py --weights <path> --runs {num_inferences} --device cpu")
    print("\n 2. When you start the command on the edge board:")
    input("    👉 Press [ENTER] HERE to START recording...")

    # Start streaming
    t0 = time.time()
    device.start()
    print("\n  🔴 RECORDING ACTIVE! Board is running inference...")
    input(f"    👉 Press [ENTER] as soon as the {num_inferences} inferences finish...")

    # Stop streaming
    device.stop()
    actual_dur = time.time() - t0
    print(f"  ⏹️ Stopped recording. Measured duration: {actual_dur:.2f}s")

    # Extract all recorded samples
    try:
        start_id, end_id = device.stream_buffer.sample_id_range
        data = device.stream_buffer.samples_get(start_id, end_id, fields=["current", "voltage"])
        i = data["signals"]["current"]["value"]
        v = data["signals"]["voltage"]["value"]
    except Exception:
        # Fallback to reading last segment if stream_buffer query fails
        data = device.read(duration=min(actual_dur, 10.0))
        if isinstance(data, dict):
            i = np.asarray(data["signals"]["current"]["value"])
            v = np.asarray(data["signals"]["voltage"]["value"])
        else:
            i = data[:, 0]
            v = data[:, 1]

    power_w = i * v
    mean_power = float(np.mean(power_w))
    mean_voltage = float(np.mean(v))
    mean_current_ma = float(np.mean(i)) * 1000.0
    total_energy_j = mean_power * actual_dur

    return {
        "duration_s": actual_dur,
        "mean_power_w": mean_power,
        "mean_voltage_v": mean_voltage,
        "mean_current_ma": mean_current_ma,
        "total_energy_j": total_energy_j,
        "num_inferences": num_inferences,
    }


def save_record(row_dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()
    
    headers = [
        "Timestamp",
        "Target_Device",
        "Model_Name",
        "Inferences_Count",
        "Total_Duration_s",
        "Avg_Latency_ms",
        "Avg_FPS",
        "Idle_Power_W",
        "Active_Power_W",
        "Dynamic_Power_W",
        "Total_Energy_J",
        "Energy_per_Frame_mJ",
        "Frames_per_Joule",
        "Mean_Voltage_V",
        "Mean_Current_mA",
    ]
    
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"  [Saved] Record appended to {OUTPUT_CSV}")


def run_single(device, target_device: str, model_name: str, weights_path: str,
               inferences: int, p_idle: float):
    print("\n" + "=" * 70)
    print(f" TARGET: {target_device} | MODEL: {model_name}")
    print(f" Command to run on {target_device}:")
    print(f"   python scripts/benchmark_edge.py --weights {weights_path} --runs {inferences} --device cpu")
    print("=" * 70)

    act = capture_active_interactive(device, model_name=model_name, num_inferences=inferences)

    p_active = act["mean_power_w"]
    total_energy_j = act["total_energy_j"]
    duration_s = act["duration_s"]

    # Calculate metrics
    dynamic_power = max(0.0, p_active - p_idle)
    energy_per_frame_j = total_energy_j / inferences
    energy_per_frame_mj = energy_per_frame_j * 1000.0
    frames_per_joule = (1.0 / energy_per_frame_j) if energy_per_frame_j > 0 else 0.0

    avg_latency_ms = (duration_s / inferences) * 1000.0
    avg_fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    # Summary table
    print("\n" + "=" * 70)
    print(f" RESULT SUMMARY ({inferences} INFERENCES)")
    print("=" * 70)
    print(f" Target Device        : {target_device}")
    print(f" Model Name           : {model_name}")
    print(f" Inferences Captured  : {inferences} frames")
    print(f" Total Duration       : {duration_s:.2f} s")
    print(f" Avg Latency / Frame  : {avg_latency_ms:.2f} ms ({avg_fps:.2f} FPS)")
    print(f" Baseline Idle Power  : {p_idle:.3f} W")
    print(f" Mean Active Power    : {p_active:.3f} W")
    print(f" Dynamic Power Delta  : {dynamic_power:.3f} W")
    print(f" Total Energy Spent   : {total_energy_j:.2f} Joules")
    print(f" Energy per Frame     : {energy_per_frame_mj:.2f} mJ / frame ⚡")
    print(f" Energy Efficiency    : {frames_per_joule:.2f} Frames / Joule")
    print("=" * 70)

    row = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Target_Device": target_device,
        "Model_Name": model_name,
        "Inferences_Count": inferences,
        "Total_Duration_s": round(duration_s, 2),
        "Avg_Latency_ms": round(avg_latency_ms, 2),
        "Avg_FPS": round(avg_fps, 2),
        "Idle_Power_W": round(p_idle, 3),
        "Active_Power_W": round(p_active, 3),
        "Dynamic_Power_W": round(dynamic_power, 3),
        "Total_Energy_J": round(total_energy_j, 3),
        "Energy_per_Frame_mJ": round(energy_per_frame_mj, 2),
        "Frames_per_Joule": round(frames_per_joule, 2),
        "Mean_Voltage_V": round(act["mean_voltage_v"], 2),
        "Mean_Current_mA": round(act["mean_current_ma"], 1),
    }
    save_record(row)
    return row


def main():
    parser = argparse.ArgumentParser(description="Joulescope Precision Energy Capture")
    parser.add_argument("--device", type=str, default="Rock Pi 4C+",
                        choices=["Rock Pi 4C+", "Raspberry Pi 4B", "Jetson AGX Xavier", "Edge Board"],
                        help="Target edge hardware name")
    parser.add_argument("--model", type=str, default="Ours (Dilated ECA v30)",
                        help="Model name for single-model run")
    parser.add_argument("--weights", type=str, default="weights/v30_champion_best.pt",
                        help="Path to model weights")
    parser.add_argument("--inferences", type=int, default=10,
                        help="Number of inferences in the test run (default: 10)")
    parser.add_argument("--suite", action="store_true",
                        help="Run multi-model suite testing all models sequentially")
    parser.add_argument("--idle-sec", type=float, default=15.0,
                        help="Seconds to sample idle baseline")
    args = parser.parse_args()

    js = check_joulescope()
    print("=" * 70)
    print(f" JOULESCOPE JS220 ENERGY BENCHMARK: {args.device}")
    print("=" * 70)

    try:
        device = js.scan_require_one(config="auto")
        device.open()
        print(f"Connected to Joulescope: {device}\n")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Joulescope: {e}")
        return

    try:
        # Step 1: Measure Idle once
        idle_info = capture_idle(device, duration_sec=args.idle_sec)
        p_idle = idle_info["idle_power_w"]

        if args.suite:
            print("\n" + "=" * 70)
            print(f" STARTING MULTI-MODEL POWER SUITE FOR {args.device}")
            print(f" {len(ALL_MODELS)} Models | {args.inferences} Inferences Each")
            print("=" * 70)

            all_results = []
            for name, path in ALL_MODELS:
                res = run_single(device, args.device, name, path, args.inferences, p_idle)
                all_results.append(res)
                print("\nModel completed. Prepare the next model on the board.")
                time.sleep(2)

            # Final Table
            print("\n" + "=" * 70)
            print(f" COMPLETE {args.device} ENERGY BENCHMARK TABLE (Table VII Ready)")
            print("=" * 70)
            print(f"{'Model':<28} | {'Power (W)':<10} | {'Latency (ms)':<12} | {'mJ/Frame':<10} | {'Frames/J':<10}")
            print("-" * 75)
            for r in all_results:
                print(f"{r['Model_Name']:<28} | {r['Active_Power_W']:<10.3f} | {r['Avg_Latency_ms']:<12.1f} | {r['Energy_per_Frame_mJ']:<10.1f} | {r['Frames_per_Joule']:<10.2f}")
            print("=" * 70)
        else:
            run_single(device, args.device, args.model, args.weights, args.inferences, p_idle)

    finally:
        device.close()
        print("\n[Joulescope] Session ended and device safely closed.")


if __name__ == "__main__":
    main()
