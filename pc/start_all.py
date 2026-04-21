import subprocess
import sys
import time
import os

def kill_zombie_processes(port=8000):
    try:
        print(f"--- Cleaning Port {port} ---")
        cmd = f"Stop-Process -Id (Get-NetTCPConnection -LocalPort {port}).OwningProcess -Force -ErrorAction SilentlyContinue"
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    except:
        pass

def main():
    kill_zombie_processes(8000)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    receiver_script = os.path.join(base_dir, "receiver.py")
    dashboard_script = os.path.join(base_dir, "dashboard.py")

    print("--- Starting PC Telemetry System ---")
    print("=========================================")
    
    # Start the FastAPI Receiver in a separate process
    receiver_proc = subprocess.Popen([sys.executable, receiver_script])
    print("OK: Receiver started on port 8000")
    
    # Give it a second to boot up
    time.sleep(2)
    
    # Start Streamlit Dashboard in a separate process
    print("OK: Starting Streamlit Dashboard...")
    dashboard_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_script])
    
    try:
        # Wait for Streamlit to exit (or User Ctrl+C)
        dashboard_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
    finally:
        # Terminate both when done
        receiver_proc.terminate()
        dashboard_proc.terminate()
        print("OK: System successfully shut down.")

if __name__ == "__main__":
    main()
