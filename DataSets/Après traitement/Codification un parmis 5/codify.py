
import re,sys

if(len(sys.argv)<3):
	print("Too few arguments")
	exit
reader = open(str(sys.argv[1]),'r')
writer = open(str(sys.argv[2]),'w')
lines = reader.readlines()
lines =lines[1:len(lines)]
for line in lines :
    code =[]
    for x in range(0,5) :
        code.append("0")
    print(code)
    code[int(line[0])-1]="1"
    prob = ""
    for x in range(0, 5):
        prob=prob+code[x]+","
    print(line[4:len(line)-1]+","+prob[:len(prob)-1])
    writer.write(line[:len(line)-1]+","+prob[:len(prob)-1]+"\n")
    
writer.close()
