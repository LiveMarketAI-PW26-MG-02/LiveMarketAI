"""MySQL connector for experiment logging."""
import mysql.connector
from contextlib import contextmanager

DB_CONFIG=dict(host="localhost",user="root",password="",database="regime_attention")

@contextmanager
def get_conn():
    conn=mysql.connector.connect(**DB_CONFIG)
    try: yield conn; conn.commit()
    finally: conn.close()

def insert_experiment(cfg):
    with get_conn() as conn:
        cur=conn.cursor()
        cur.execute("INSERT INTO experiments (run_name,d_model,n_regimes,n_sources,lr,epochs,seed) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    ("run",cfg.d_model,cfg.n_regimes,cfg.n_sources,cfg.lr,cfg.epochs,cfg.seed))
        return cur.lastrowid

def log_epoch(exp_id,epoch,train_loss,val_loss):
    with get_conn() as conn:
        cur=conn.cursor()
        cur.execute("INSERT INTO epoch_metrics (experiment_id,epoch,train_loss,val_loss) VALUES (%s,%s,%s,%s)",(exp_id,epoch,train_loss,val_loss))
