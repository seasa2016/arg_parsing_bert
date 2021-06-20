import os

f_out = open('result.txt', 'w')

for key in os.listdir('./compare'):
    with open('./compare/'+key+'/eval_results.txt') as f:
        f_out.write(''+key+':\n')

        for line in f:
            f_out.write(line)

f_out.close()


