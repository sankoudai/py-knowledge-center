global_var = 1

def mod_op():
    global global_var
    global_var = 0

if __name__ == '__main__':
    mod_op()
    print (global_var)