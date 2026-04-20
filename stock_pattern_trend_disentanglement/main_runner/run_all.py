import os
folders = ['novelty1','novelty2','novelty3','novelty4','novelty5','novelty6','novelty7','novelty8','novelty9','novelty10']
for f in folders:
    print("Running", f)
    os.system(f"python {f}/module.py")
