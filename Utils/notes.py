def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result


    #Concat gives the flexibility to join based on the axis( all rows or all columns)
    #Append is the specific case(axis=0, join='outer') of concat
    #Join is based on the indexes (set by set_index) on how variable =['left','right','inner','couter']
    #Merge is based on any particular column each of the two dataframes, this columns are variables on like 'left_on', 'right_on', 'on'



""""
pandas.DataFrame
Display number of rows, columns, etc.: df.info()
Get the number of rows: len(df)
Get the number of columns: len(df.columns)
Get the number of rows and columns: df.shape
Get the number of elements: df.size
name of the colums: df. columns
Notes when specifying index
""""