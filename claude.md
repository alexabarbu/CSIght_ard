# Wi-Fi Motion Sensing – hackathon project

We turn ordinary Wi-Fi into a camera-free motion sensor. An ESP32-C3 connects to a Wi-Fi hotspot run by an Arduino UNO Q and pings it constantly. When a person moves near the link, the channel state information (CSI) of the reply packets changes. We read the CSI on a laptop, compute a motion score, classify the room as **empty / standing / walking**, show it on a live dashboard and trigger an alarm on the UNO Q.

Hackathon: Junction, Fri 13 Nov 19:30 – Sun 15 Nov 10:00 (about 38 hours). Team of 3. Keep everything simple and demo-ready.

## Hardware and network

```
 ESP32-C3  ──── Wi-Fi (2.4 GHz, fixed channel) ────  Arduino UNO Q  (hotspot / AP)
    │ USB-C (serial, CSI lines)                          ▲
    ▼                                                    │ Wi-Fi, same network (alarm messages)
 Laptop (macOS) ─────────────────────────────────────────┘
```

- 1× ESP32-C3 dev board (the sensor), connected to the laptop over USB-C. Connects to the UNO Q hotspot and pings it ~100×/s; each reply gives one CSI measurement.
- 1× Arduino UNO Q (Debian Linux + Python): **the Wi-Fi hotspot** (2.4 GHz, fixed channel, created with `nmcli`) **and** the alarm (LED matrix or buzzer). Its Wi-Fi (ath10k driver) cannot give CSI itself, but it can be the access point. Keep it still: moving the UNO Q changes the CSI.
- Laptop (macOS) running all Python code. Joins the UNO Q hotspot so it can send alarm messages to it. While joined, the laptop has no internet over Wi-Fi unless the UNO Q shares an uplink, so install packages beforehand.

Network defaults (change in one place if different):
- SSID / password: set in `nmcli` on the UNO Q and in `firmware/sdkconfig.defaults`
- Channel: fixed (e.g. 6), band `bg` (2.4 GHz)
- UNO Q address on the hotspot: `10.42.0.1` (nmcli `ipv4.method shared` default). This is both the ESP32's ping target (its gateway) and the dashboard's alarm address.

## Folder layout

```
junction/
├── CLAUDE.md
├── firmware/          ← ESP-IDF project, adapted from Espressif esp-csi csi_recv_router (see firmware/README.md)
├── read_csi.py        ← reads CSI lines from USB, turns them into amplitudes
├── features.py        ← 1 second of amplitudes → feature numbers (motion score etc.)
├── record.py          ← records labelled training data to CSV
├── train.py           ← trains the classifier, saves model.pkl
├── dashboard.py       ← live status + heatmap, sends the alarm
├── uno_q/             ← everything on the UNO Q: hotspot setup script + alarm server
├── data/              ← recorded CSVs (one per label)
└── requirements.txt
```

## Build order (do the steps in this order; each has a "done when")

### 0. Hotspot on the UNO Q
Create a 2.4 GHz access point on a fixed channel with `nmcli` (keep it as a script in `uno_q/setup_hotspot.sh`):
```sh
nmcli con add type wifi ifname wlan0 con-name csi-ap autoconnect yes ssid <SSID> \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  802-11-wireless.powersave 2 \
  ipv4.method shared wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<password>"
nmcli con up csi-ap
```
`powersave 2` turns Wi-Fi power saving off on the UNO Q, so replies aren't delayed.
**Done when:** the laptop can join the network and `ping 10.42.0.1` works.

### 1. Firmware (ESP32-C3) – done, in `firmware/`
Adapted from Espressif's `esp-csi` `csi_recv_router`. Changes from the example: Wi-Fi power save off, CSI printed from a separate task (not the Wi-Fi callback), output on both UART and native USB, and a `STATUS,...` line every 5 s.
Set the UNO Q hotspot SSID/password in `firmware/sdkconfig.defaults`, delete `firmware/sdkconfig` if it exists, then `idf.py -p <port> flash monitor`.
Serial: **921600 baud**. One line per packet:
`CSI_DATA,seq,mac,rssi,rate,...,len,first_word,"[v0,v1,...]"` (128 values = 64 subcarriers × (imag, real)).
**Done when:** the monitor shows a stream of `CSI_DATA` lines and `STATUS,csi_per_s=` is around 90–100.

### 2. `read_csi.py`
- Open the serial port with `pyserial` (port and baud rate as command-line arguments; default 921600).
- Read lines, keep only those starting with `CSI_DATA` (ignore `STATUS`, logs and partial lines).
- Parse the CSI number list at the end of the line. **Check the parser against a real captured line.**
- The numbers come in pairs per subcarrier. Amplitude per subcarrier = sqrt(a² + b²). Ignore phase. Drop the null/guard subcarriers.
- Expose a generator/function that yields `(timestamp, numpy array of amplitudes)` so other scripts can reuse it.
- Skip malformed lines without crashing.
**Done when:** running it prints one row of ~50 amplitudes per packet.

### 3. `features.py`
- Collect a sliding window of the last ~1 second of amplitude rows.
- Features per window: motion score (std of each subcarrier over the window, averaged), plus mean, max, min and a few percentiles of that std, and the mean amplitude.
- Simple function: `extract_features(window) -> dict`.
- Small CLI that prints the motion score live, plus a simple threshold detector (`MOTION` / `NO MOTION`).
**Done when:** the score is low in an empty room and clearly jumps when someone walks between the UNO Q and the ESP32.

### 4. `record.py`
- Usage: `python record.py <label> --seconds 180` (labels: `empty`, `standing`, `walking`).
- Saves one feature row per window to `data/<label>.csv` (append).
**Done when:** there is one CSV per label with a few minutes of data each.

### 5. `train.py`
- Load all CSVs in `data/`, train a scikit-learn `RandomForestClassifier`.
- Print accuracy and a confusion matrix on a held-out split.
- Save the model as `model.pkl` (joblib).
**Done when:** accuracy is reasonable and `model.pkl` exists.

### 6. `dashboard.py`
- Streamlit app. Reads CSI live (reuse `read_csi.py` + `features.py`), loads `model.pkl`.
- Shows: current status (empty / standing / walking), confidence, live motion score chart, live heatmap of amplitudes (time × subcarrier).
- Falls back to the threshold detector if `model.pkl` is missing.
- When status is `walking` (or motion above threshold), send an alarm message to the UNO Q over the hotspot network (simple HTTP request or TCP message; address configurable, default `10.42.0.1`).
- Tamper alert: if no CSI packets arrive for a few seconds, show "Sensor link lost".
**Done when:** walking past changes the screen within 1–2 seconds.

### 7. `uno_q/` alarm server
- Small Python server on the UNO Q's Linux side, listening on the hotspot interface. It receives the alarm message and lights the LED matrix or a buzzer (via the board's microcontroller side).
**Done when:** walking past makes the UNO Q light up.

## Priorities
- Steps 0–3 are the must-have core (working motion detector).
- Steps 4–6 are the main goal.
- Step 7 is a nice extra.
- Stretch only if everything works: local AI chat over presence data, running the pipeline on the UNO Q or a Jetson.

## Coding rules
- Python 3, keep dependencies minimal: `pyserial`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `streamlit`. Keep `requirements.txt` up to date.
- Configuration (serial port, baud rate, window length, threshold, UNO Q address) via command-line arguments or a small config section at the top, not hard-coded deep in the code.
- Short, readable functions with comments explaining what each step does; the team is learning signal processing as it goes.
- Never crash on a bad serial line; log and continue.
- Everything runs locally; no cloud services, no data leaves the laptop.
