import subprocess
import sys
 
def gpu_status():
  gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader"])#,noheader,nounits"])
  gpu_info = gpu_info.decode('UTF-8')
  print("gpu_info:", gpu_info, end = '')
  sys.stdout.flush()

gpu_status()