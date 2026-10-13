# import cityflow
#
# con='/home/dl/PycharmProjects/sumo-cityflow/3_3/test.json'
# eng = cityflow.Engine(con)
#
#
# a=eng.get_lane_vehicle_count()
# b=eng.get_lane_vehicles()
# intersction='intersection_1_1'
# # c=eng.get_vehicles(include_waiting=False)
# D=eng.get_vehicles(include_waiting=True)
#
# #相位
# for i in range (1000):
#     # if i in range(0,200):
#     eng.set_tl_phase(intersction,1)
#     eng.next_step()
    # if i in range(200,400):
    #     eng.set_tl_phase(intersction, 1)
    #     eng.next_step()
    # if i in range(400,600):
    #     eng.set_tl_phase(intersction, 2)
    #     eng.next_step()
    # if i in range(600,800):
    #     eng.set_tl_phase(intersction, 3)
    #     eng.next_step()

#
# a={"01":{"11":{"21":1,"22":2},"12":{"21":1,"22":2}},"02":{"11":{"21":1,"22":2},"12":{"21":1,"22":2}}}
#
# for k1,v1 in a.items():
#     for k2,v2 in v1.items():
#         for k3,v3 in v2.items():
#             a[k1][k2][k3]=555
#
import os
#
# a=os.path.dirname("roadnet_3_4")
# print(os.path.abspath('.')  )
# # print(path2)

ab_path = os.path.abspath('..')

prefix_intersections = str("3_4")
PATH_TO_DATA = os.path.join(ab_path, prefix_intersections)
print(PATH_TO_DATA)