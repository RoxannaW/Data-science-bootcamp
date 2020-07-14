import pandas as pd
import numpy as np


"""functiones to clean strings"""
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result


"""" To read all line of a json file. """"
with open('name_file') as f:
    line = f.readline()
    if line:
        print(line)
        line = f.readline()
print(line)


""" Lambda function to transform to float and not take into account first elemen (dollar sign in most
 cases,but change how needed."""

Floater = lambda x: float(x[1:-1])


