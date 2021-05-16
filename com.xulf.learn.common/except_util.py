def assert_except(f):
    try:
        f()
        assert False
    except Exception as e:
        assert True


def assert_no_except(f):
    try:
        f()
        assert True
    except Exception as e:
        assert False