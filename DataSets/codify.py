reader = open("dataSetHandPosture.csv",'r')
writer = open("dataOneOf5.csv",'w')

lines = reader.readlines()
writer.writelines(lines[0][0:len(lines[0]) - 2] + ",a,w,i,s,\n")
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