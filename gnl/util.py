from functools import wraps

def argkws(f):
    """Function decorator which maps f((args, kwargs)) to f(*args, **kwargs)"""
    @wraps(f)
    def fun(tup):
        args, kws = tup
        return fun(*args, **kw)

