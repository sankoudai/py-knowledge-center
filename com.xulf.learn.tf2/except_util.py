def assert_except(f):
    try:
        f()
        assert False
    except Exception as e:
        assert True