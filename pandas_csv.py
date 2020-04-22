import pandas
io = pandas.read_csv('Transposed_pixels.csv',sep=",",usecols=(1,2,4)) # To read 1st,2nd and 4th columns
print (io) 