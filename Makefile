# Shortcuts for the ESP32-C3. Run from the repo root, e.g. "make flash".
# Stop the monitor: Ctrl+T then Ctrl+X. Stop "make read": Ctrl+C.

PORT ?= /dev/cu.usbmodem101
IDF  = . ~/esp/esp-idf/export.sh > /dev/null && cd firmware && idf.py -p $(PORT)

SHELL := /bin/bash

# Build, upload and show the output (only needed after changing the code)
flash:
	$(IDF) flash monitor

# Show the output without uploading again
monitor:
	$(IDF) monitor

# Print the raw lines, no ESP-IDF needed
read:
	cat $(PORT)

# Stop whatever is holding the port
kill:
	-kill $$(lsof -t $(PORT)) 2>/dev/null

# Delete the build folder (fixes "build directory configured for another project")
clean:
	rm -rf firmware/build

# --- Recording data (saved in data/, a new numbered file per run) ---
PY = .venv/bin/python

# One time only: create the Python environment
setup:
	python3 -m venv .venv && $(PY) -m pip install pyserial numpy

# 1. Empty room (you sit still, as far from the line ESP32 <-> phone as possible)
empty:
	$(PY) record.py empty 30

# 2. One person standing still. Run once per position.
standing:
	$(PY) record.py standing 20

# 3. One person walking around. Run 2-3 times.
walking:
	$(PY) record.py walking 20

.PHONY: flash monitor read kill clean setup empty standing walking
