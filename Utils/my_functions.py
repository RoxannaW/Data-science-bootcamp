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



"""function to change gender column to numeric, to then use to calculate ratios"""
def gender_to_numeric(x):
    if x == 'M':
        return 1
    if x == 'F':
        return 0

# apply the function to the gender column and create a new column
df['new_column_name'] = df['gender_column'].apply(gender_to_numeric)
# use the new column and sum via groupby specific column and the vulue count of the specific column to calculate ratio
variabel = (df.groupby('specific_column').new_column_name.sum() / df.specific_column.value_counts()) * 100 
# add % sign
variabel.astype(str) + "%"



""" lambda function to show all the columns and values after doing a groupby """
lambda df:df.sort_values(by=["column_name1","column_name2"]) # can also do only one column

#use as following: 
#variabel = name_df.groupby(["column_name1", "column_name2"])
#variabel = variabel.apply("insert Lambda function")
#delete the duplicate columns names


"""function to replace values of a column that are negative with the mean value 
    of the column (not taking into account the negative values)"""

def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group

## use as following: new_info = df.groupby("columns name of groupby")["column to take into account"].apply(replace)


"""Clean strings:"""
df["Column"].str.elem.extract('([a-zA-Z\s]+)', expand=False).str.strip()