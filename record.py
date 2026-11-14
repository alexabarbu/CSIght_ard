# Records CSI from the ESP32 into data/<label>_<n>.csv
# Usage: python record.py <label> <seconds>     e.g. python record.py walking 20
# Each run gets a new file: walking_1.csv, walking_2.csv, ...

import os, sys, time
import serial

PORT = "/dev/cu.usbmodem101"
label, seconds = sys.argv[1], int(sys.argv[2])

# Pick the next free file name
os.makedirs("data", exist_ok=True)
n = 1
while os.path.exists(f"data/{label}_{n}.csv"):
    n += 1
path = f"data/{label}_{n}.csv"

# Open the port without toggling DTR/RTS (that would reset the ESP32)
ser = serial.Serial()
ser.port, ser.baudrate, ser.timeout = PORT, 115200, 1
ser.dtr = ser.rts = False
ser.open()

for i in range(5, 0, -1):
    print(f"Get in position... {i}")
    time.sleep(1)
ser.reset_input_buffer()  # throw away what arrived during the countdown
print(f"RECORDING '{label}' for {seconds} s -> {path}")

count = 0
start = time.time()
with open(path, "w") as f:
    f.write("time,rssi," + ",".join(f"v{i}" for i in range(128)) + "\n")
    while time.time() - start < seconds:
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("CSI_DATA"):
            continue
        # CSI_DATA,<rssi>,<len>,[v0,...,v127]
        _, rssi, _, values = line.split(",", 3)
        values = values.strip("[]")
        if values.count(",") != 127:  # broken line, skip it
            continue
        f.write(f"{time.time() - start:.3f},{rssi},{values}\n")
        count += 1

print(f"Done: {count} packets (~{count / seconds:.0f} per second) saved to {path}")
if count == 0:
    print("No CSI received! Is the hotspot on and the ESP32 connected? Try 'make read'.")
