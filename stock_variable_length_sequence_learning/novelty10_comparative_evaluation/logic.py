def main_logic(data):
    fixed = data[-5:]
    variable = data[-(len(data)//2):]
    return {
        "fixed_avg": sum(fixed)/len(fixed),
        "variable_avg": sum(variable)/len(variable)
    }
