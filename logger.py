import serial
import csv
from datetime import datetime

# 🔧 CHANGE THIS to your port
ser = serial.Serial('COM7', 115200, timeout=1)

file_name = "new.csv"

with open(file_name, 'a', newline='') as file:
    writer = csv.writer(file)

    # Write header if file is empty
    writer.writerow(["Timestamp", "MQ2", "MQ3", "MQ135", "Temperature", "Humidity"])

    print("Logging started...")

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()

            if not line:
                continue

            # Skip header or errors
            if "MQ2" in line or "ERROR" in line:
                continue

            values = line.split(',')

            if len(values) == 5:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                row = [timestamp] + values
                writer.writerow(row)
                file.flush()

                print("Saved:", row)

        except Exception as e:
            print("Error:", e)