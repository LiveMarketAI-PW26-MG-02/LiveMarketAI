from utils import volatility
from config import WINDOW_MIN, WINDOW_MAX

def main_logic(data):
    vol = volatility(data)
    k = int(min(WINDOW_MAX, max(WINDOW_MIN, len(data) * (1 + vol))))
    return data[-k:]
