
import os
folders = [f"novelty{i}" for i in range(1,11)]
for f in folders:
    print("Running", f)
    os.system(f"python {f}/script.py")
