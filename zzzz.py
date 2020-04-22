import itertools
a = [1,2,3,4,5]
b = [6,7,8,9]
for f in a and b:
	print(f)
s = [ n for n in itertools.chain(a,b) ]
print(s)