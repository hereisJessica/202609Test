import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


Y_value=[]
X_value=[]

r_n=180
l_n=1


def AeragList(list,count):
    newlis=[]
    #each_r=0
    for i in range(len(list)):
        if i!=0 and i % count == 0:
            newlis.append(averagenum(list[i-count:i]))
            #each_r += list[1]
    return newlis

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)




def finevalue(list):
    Y_value = []
    r=0
    for value in list :
        Y_value.append(value)
        r+=value

    return Y_value,r/len(list)


def rewardvalue(list):
    Y_value = []
    for value in list:
        Y_value.append(value[2])
    return Y_value


#获得每个agent的reward
def findvalue(list,agent):

    Y_value = []
    if agent == 1:
        for value in list :
            Y_value.append(value[1])
    if agent == 2:
        for value in list :
            Y_value.append(value[2])
    if agent == 3:
        for value in list:
            Y_value.append(value[3])
    if agent == 4:
        for value in list :
            Y_value.append(value[4])
    if agent == 5:
        for value in list:
            Y_value.append(value[5])
    if agent == 6:
        for value in list:
            Y_value.append(value[6])
    if agent == 7:
        for value in list:
            Y_value.append(value[7])
    if agent == 8:
        for value in list:
            Y_value.append(value[8])
    if agent == 9:
        for value in list:
            Y_value.append(value[9])

    return Y_value

#获得loss
def findloss(list,a_or_l):

    Y_value = []
    if a_or_l == 1:
        for value in list :
            Y_value.append(value[1])
    if a_or_l == 2:
        for value in list :
            Y_value.append(value[2])

    return Y_value

#平均值
def average(list_y):
    total=0
    for i in range (len(list_y)):
        total+=list_y[i]
    return total/len(list_y)


'''
reward
'''
A1_R=[]
A2_R=[]
A3_R=[]
A4_R=[]
A5_R=[]
A6_R=[]
A7_R=[]
A8_R=[]
A9_R=[]
all_r=[]

reward_data=pd.read_csv('MPLight/MARDDPG_reward_train__2022_01_20-10_21/each_reward_2022_01_20-10_21')
for row in reward_data.values.tolist():
    A1_R.append(row[1])
    A2_R.append(row[2])
    A3_R.append(row[3])
    A4_R.append(row[4])
    A5_R.append(row[5])
    A6_R.append(row[6])
    A7_R.append(row[7])
    A8_R.append(row[8])
    A9_R.append(row[9])
    a=row[1:]
    all_r.append((np.sum(row[1:]))/9)

A1_R = AeragList(A1_R, r_n)
A2_R = AeragList(A2_R, r_n)
A3_R = AeragList(A3_R, r_n)
A4_R = AeragList(A4_R, r_n)
A5_R = AeragList(A5_R, r_n)
A6_R = AeragList(A6_R, r_n)
A7_R = AeragList(A7_R, r_n)
A8_R = AeragList(A8_R, r_n)
A9_R = AeragList(A9_R, r_n)
all_reward=AeragList(all_r, r_n)

# plt.plot(A1_R,color='red')
# plt.plot(A2_R,color='red')
# plt.plot(A3_R,color='red')
# plt.plot(A4_R,color='red')
# plt.plot(A5_R,color='red')
# plt.plot(A6_R,color='red')
# plt.plot(A7_R,color='red')
# plt.plot(A8_R,color='red')
# plt.plot(A9_R,color='red')
plt.plot(all_reward,color='red')
plt.ylabel("REWARD")
plt.show()

#
# '''
# --------------COUNT----------------------------------------------------------------
# '''
# A1_N=[]
# A2_N=[]
# A3_N=[]
# A4_N=[]
# A5_N=[]
# A6_N=[]
# A7_N=[]
# A8_N=[]
# A9_N=[]
# all_N=[]
#
# N_data=pd.read_csv('./reward/each_count.csv')
# for row in N_data.values.tolist():
#     A1_N.append(row[1])
#     A2_N.append(row[2])
#     A3_N.append(row[3])
#     A4_N.append(row[4])
#     A5_N.append(row[5])
#     A6_N.append(row[6])
#     A7_N.append(row[7])
#     A8_N.append(row[8])
#     A9_N.append(row[9])
#     a=row[1:]
#     all_N.append((np.sum(row[1:]))/9)
#
# A1_N = AeragList(A1_N, r_n)
# A2_N = AeragList(A2_N, r_n)
# A3_N= AeragList(A3_N, r_n)
# A4_N = AeragList(A4_N, r_n)
# A5_N = AeragList(A5_N, r_n)
# A6_N = AeragList(A6_N, r_n)
# A7_N = AeragList(A7_N, r_n)
# A8_N = AeragList(A8_N, r_n)
# A9_N = AeragList(A9_N, r_n)
# all_N= AeragList(all_N, r_n)
#
# # plt.plot(A1_N,color='red')
# # plt.plot(A2_N,color='red')
# # plt.plot(A3_N,color='red')
# # plt.plot(A4_N,color='red')
# # plt.plot(A5_N,color='red')
# # plt.plot(A6_N,color='red')
# # plt.plot(A7_N,color='red')
# # plt.plot(A8_N,color='red')
# # plt.plot(A9_N,color='red')
# plt.plot(all_N,color='red')
# plt.ylabel("total N")
# plt.show()
#
# A1_W=[]
# A2_W=[]
# A3_W=[]
# A4_W=[]
# A5_W=[]
# A6_W=[]
# A7_W=[]
# A8_W=[]
# A9_W=[]
# all_W=[]
#
# w_data=pd.read_csv('./reward/each_wait.csv')
# for row in w_data.values.tolist():
#     A1_W.append(row[1])
#     A2_W.append(row[2])
#     A3_W.append(row[3])
#     A4_W.append(row[4])
#     A5_W.append(row[5])
#     A6_W.append(row[6])
#     A7_W.append(row[7])
#     A8_W.append(row[8])
#     A9_W.append(row[9])
#     a=row[1:]
#     all_W.append((np.sum(row[1:]))/9)
#
# A1_W = AeragList(A1_W, r_n)
# A2_W = AeragList(A2_W, r_n)
# A3_W= AeragList(A3_W, r_n)
# A4_W = AeragList(A4_W, r_n)
# A5_W = AeragList(A5_W, r_n)
# A6_W = AeragList(A6_W, r_n)
# A7_W = AeragList(A7_W, r_n)
# A8_W = AeragList(A8_W, r_n)
# A9_W = AeragList(A9_W, r_n)
# all_W= AeragList(all_W, r_n)
#
# # plt.plot(A1_W,color='red')
# # plt.plot(A2_W,color='red')
# # plt.plot(A3_W,color='red')
# # plt.plot(A4_W,color='red')
# # plt.plot(A5_W,color='red')
# # plt.plot(A6_W,color='red')
# # plt.plot(A7_W,color='red')
# # plt.plot(A8_W,color='red')
# # plt.plot(A9_W,color='red')
# plt.plot(all_W,color='red')
# plt.ylabel("Wait_time")
# plt.show()