def main_logic(data):
    short = data[-3:]
    mid = data[-5:]
    long = data
    return {"short": sum(short)/len(short), "mid": sum(mid)/len(mid), "long": sum(long)/len(long)}
