"""Centralised experiment config."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class AttentionConfig:
    d_model:int=256; n_regimes:int=4; n_sources:int=8; n_heads:int=8
    dropout:float=0.1; lr:float=3e-4; weight_decay:float=1e-4
    batch_size:int=64; epochs:int=50; seed:int=42; entropy_coeff:float=0.01
    device:str="cuda"; log_dir:str="runs/"
    regime_names:List[str]=field(default_factory=lambda:["trending","volatile","mixed","mean_reverting"])
    def to_dict(self):
        import dataclasses; return dataclasses.asdict(self)
