import os
import pathlib
import sys
import pandas
import numpy as np

def exludeUser(values,userId):
    leftUser = []
    otherUsers = []
    for val in values:
        if (val[1] == userId):
            leftUser.append(val)
        else:
            otherUsers.append(val)
    return np.array(leftUser),np.array(otherUsers)


if(len(sys.argv)<2):
	print("Too few argumente")
	exit


#data = pandas.read_csv(sys.argv[1])
data = pandas.read_csv(sys.argv[1]).replace('?', 0)

for user in range(0,15):
    leftUser, values = exludeUser(data.values, user)
    if(len(leftUser))==0:
        print(user)
        continue
    others = pandas.DataFrame(data=values,index=values[0:,0],columns=values[0,0:])
    # print(leftUser)
    # print("///////////////////")
    # print(values)
    left = pandas.DataFrame(data=leftUser,
                            index=leftUser[0:,0],
                            columns=leftUser[0,0:])
    pathlib.Path('usersLeft/user'+str(user)).mkdir(parents=True, exist_ok=True)
    others.to_csv("usersLeft/user"+str(user)+"/Training.csv", sep=',',index=False)
    left.to_csv("usersLeft/user"+str(user)+"/Testing.csv", sep=',',index=False)

    # print(left.values)
