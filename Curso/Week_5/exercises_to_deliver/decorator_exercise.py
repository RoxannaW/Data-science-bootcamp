import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps


def prepost(*arg, **kwargs):
    def inner(function):
        @wraps(function)
        def wrapper(*args_function, **kwargs_function):
            if "url" in kwargs:
                df = pd.read_csv(list(kwargs.values())[0])
                result = function(*args_function, **kwargs_function)
                df.hist()
                plt.show()
            return result
        return wrapper
    return inner
            

@prepost(url='http://winterolympicsmedals.com/medals.csv')
def _f_protected_1():
    l1 = [ x for x in range(16)]
    list_1 = list(filter(lambda x: True if x > 5 else False, l1))
    print("hello")
    print(list_1)
    return list_1
        
_f_protected_1()