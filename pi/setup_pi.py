import os
import subprocess
import sys

def run_command(command, use_sudo=False):
    if use_sudo:
        command = f"sudo {command}"
    print(f"\n[EXEC] Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error Code: {e.returncode}")
        sys.exit(1)

def main():
    print("==============================================")
    print("🍊 Raspberry Pi Setup Script for Orange Scanner")
    print("==============================================")
    print("This script will install system dependencies for OpenCV and install required Python modules.")
    
    # 1. Update APT
    print("\n>>> Updating system package list...")
    run_command("apt-get update", use_sudo=True)
    
    # 2. Install System Dependencies for OpenCV (cv2)
    # The Pi often fails to run cv2 without these native C++ libraries
    print("\n>>> Installing Native Dependencies for OpenCV...")
    # Using python3-opencv from apt is the safest way to get cv2 working on a Pi
    apt_deps = [
        "python3-opencv",
        "libgl1",
        "libopenblas-dev",
        "libhdf5-dev",
        "libqt5gui5",
        "python3-venv"
    ]
    apt_command = "apt-get install -y " + " ".join(apt_deps)
    run_command(apt_command, use_sudo=True)
    
    # 3. Create a Virtual Environment (Mandatory for newer Raspberry Pi OS)
    print("\n>>> Setting up Python Virtual Environment (Inheriting system packages)...")
    venv_dir = "pi_env"
    if not os.path.exists(venv_dir):
        run_command(f"python3 -m venv --system-site-packages {venv_dir}")
    
    # Prefix pip to use the virtual environment's pip
    pip_exec = f"{venv_dir}/bin/pip"
    
    # Upgrade pip
    run_command(f"{pip_exec} install --upgrade pip")

    # 4. Install requirements.txt using the virtual environment
    print("\n>>> Installing Python Libraries...")
    req_path = "requirements_pi.txt"
    if os.path.exists(req_path):
        run_command(f"{pip_exec} install -r {req_path}")
    else:
        print("⚠️ requirements_pi.txt not found, installing directly...")
        # Fallback installs
        run_command(f"{pip_exec} install numpy opencv-python tflite-runtime picamera2 requests")

    print("\n==============================================")
    print("✅ Setup Complete!")
    print("==============================================")
    print("To run your scanner, you MUST activate the environment first:")
    print("1. Run:  source pi_env/bin/activate")
    print("2. Run:  python run_scanner.py")
    print("==============================================\n")

if __name__ == "__main__":
    main()
