"""Experiment logger."""
import logging, json
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    def __init__(self, run_name, log_dir="runs"):
        self.run_name=run_name; self.log_dir=Path(log_dir)/run_name
        self.log_dir.mkdir(parents=True,exist_ok=True); self.metrics=[]
        logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
            handlers=[logging.FileHandler(self.log_dir/"train.log"),logging.StreamHandler()])
        self.log=logging.getLogger(run_name)
    def log_epoch(self,epoch,**kwargs):
        r={"epoch":epoch,"ts":datetime.utcnow().isoformat(),**kwargs}
        self.metrics.append(r); self.log.info(json.dumps(r))
    def save(self):
        with open(self.log_dir/"metrics.json","w") as f: json.dump(self.metrics,f,indent=2)
