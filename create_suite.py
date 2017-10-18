import random

f = open("test_views.py")

s = f.read()

t = s.split("class")

start = t[0]
t = t[1:]

for i in range(len(t)):
	t[i] = "class" + t[i]

xs = []

c = 0
while c < 100:
	xs.append(random.sample(t,6))
	fname = "test_suite_"+str(c)+".py"
	fout = open(fname, 'w')
	fout.write(start)
	for x in xs[c]:
		fout.write(x)
	c+=1