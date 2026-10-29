"""
scripts/joulescope_power_keepalive.py
Keeps Joulescope JS220 power output continuously active so the connected
edge board stays powered ON, boots up, and stays online.
Auto-reconnects if USB is briefly bumped or re-plugged.
"""

import time
import sys
import numpy as np
import joulescope

def main():
    print("=" * 60)
    print("  JOULESCOPE JS220 POWER KEEPALIVE & LIVE MONITOR")
    print("=" * 60)

    while True:
        device = None
        try:
            devices = joulescope.scan(config="auto")
            if not devices:
                print("\r  [Waiting for Joulescope JS220 USB connection...]   ", end="", flush=True)
                time.sleep(1.0)
                continue
            device = devices[0]
            device.open()
            device.parameter_set("i_range", "auto")
            device.parameter_set("sensor_power", "on")
            print(f"\n  Device: {device}")
            print("  Status: Power output is LOCKED ON (i_range=auto)")
            print("  Press Ctrl+C to terminate and turn off power.\n")

            while True:
                data = device.read(duration=0.5)
                v = float(np.mean(np.abs(data[:, 1])))
                i = float(np.mean(np.abs(data[:, 0]))) * 1000.0
                p = (v * i) / 1000.0
                print(f"\r  ⚡ Power: {p:.3f} W | Voltage: {v:.2f} V | Current: {i:.1f} mA   ", end="", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n\nShutting down power keepalive...")
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
            break
        except Exception as e:
            print(f"\n  [USB glitch/reconnect: {e}] Reconnecting in 1s...")
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
            time.sleep(1.0)

if __name__ == "__main__":
    main()
