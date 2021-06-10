import multiprocessing as mp


def work_func(x, output):
    output.put(x * x)

def test_func():
    # Define an output queue
    output = mp.Queue()

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=work_func, args=(x, output)) for x in range(4)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]

    print(results)

if __name__ == '__main__':
    test_func()
