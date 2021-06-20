from sklearn.metrics import f1_score
import sys

data = []

in_data = [None,None]
with open(sys.argv[1]) as f:
	for i,line in enumerate(f):
		line = line.strip().split()
		in_data[i%2] = line

		if(i%2==1):
			data.append( f1_score(y_true=in_data[1], y_pred=in_data[0],average='macro'))

print(sum(data)/len(data))

