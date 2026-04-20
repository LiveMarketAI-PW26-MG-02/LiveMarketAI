def main_logic(data):
    short = data[-max(3, len(data)//4):]
    long = data
    return {"short_avg": sum(short)/len(short), "long_avg": sum(long)/len(long)}
