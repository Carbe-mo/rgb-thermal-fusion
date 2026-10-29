"""
scripts/joulescope_power_keepalive.py
Keeps Joulescope JS220 power output continuously active so the connected
edge board (Raspberry Pi 5) stays powered ON, boots up, and stays online.
Also monitors and displays real-time voltage, current, and power.
"""

import time
import sys
import numpy as np
import joulescope

def main():
    print("=" * 60)
    print("  JOULESCOPE JS220 POWER KEEPALIVE & LIVE MONITOR")
    print("=" * 60)

    device = joulescope.scan_require_one(config="auto")
    device.open()
    
    # Enable continuous power output
    device.parameter_set("i_range", "auto")
    device.parameter_set("sensor_power", "on")
    print(f"  Device: {device}")
    print("  Status: Power output is LOCKED ON (i_range=auto)")
    print("  Press Ctrl+C to terminate and turn off power.\n")

    try:
        while True:
            data = device.read(duration=0.5)
            v = np.mean(np.abs(data[:, 1]))
            i = np.mean(np.abs(data[:, 0])) * 1000.0
            p = (v * i) / 1000.0
            print(f"\r  ⚡ Power: {p:.3f} W | Voltage: {v:.2f} V | Current: {i:.1f} mA   ", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nShutting down power keepalive...")
    finally:
        try:
            device.close()
        except Exception:
            pass
        print("Joulescope closed.")

if __name__ == "__main__":
    main()
