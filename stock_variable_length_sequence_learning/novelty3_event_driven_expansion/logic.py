from utils import event_magnitude
from config import EVENT_THRESHOLD

def main_logic(data):
    if event_magnitude(data) > EVENT_THRESHOLD:
        return data  # expand to full
    return data[-max(3, len(data)//3):]  # contract
