import json
import argparse


'''
参数解析
scenario:景
num_step:运行步数
'''
import os
def parse_arguments():
    parser = argparse.ArgumentParser()
    dir= os.path.abspath('..')
    road_dir=os.path.join(dir,'3_4')


    parser.add_argument("--scenario", type=str, default="3_4")
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--road_net",   type=str, default='3_4')
    parser.add_argument("--data_temp", type=str, default="jn")
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--alg",type=str,default='MPLight')
    parser.add_argument("--test",type=str,default='_')




    return parser.parse_args()




import sys
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')
        log_dir = os.path.dirname(filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def write(self, message):
            if self.add_flag:
                with open(self.filename, 'a+') as log:
                    self.terminal.write(message)
                    log.write(message)
            else:
                with open(self.filename, 'w') as log:
                    self.terminal.write(message)
                    log.write(message)

'''
路网解析
'''

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict ={}

    '''
    获取到每个路口的车道的状态就可以了
   '''
    intersection_road={}

    road={}
    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    for intersection in roadnet["intersections"]:

        phase = {}
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                     "end_lane": [],
                                                     "phase": [],
                                                     "phase_startLane_mapping": {},
                                                     "phase_roadLink_mapping": {},
                                                     }
        road_links = intersection["roadLinks"]
        road_ew=intersection['roads'][0:4:2]
        road_sn = intersection['roads'][1:4:2]
        road=road_ew+road_sn
        i=intersection['id']
        phase['PA']=road_ew
        phase['PC']=road_sn
        phase['PB']=road_ew
        phase['PD']=road_sn

        intersection_road[i]=phase

        start_lane = []
        end_lane = []
        roadLink_lane_pair = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                end_lane.append(el)
                roadLink_lane_pair[ri].append((sl, el))

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))

        for phase_i in range(1, len(intersection["trafficLight"]["lightphases"])):
            # a=len(intersection["trafficLight"]["lightphases"])
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            for ri in p["availableRoadLinks"]:
                lane_pair.extend(roadLink_lane_pair[ri])
                if roadLink_lane_pair[ri][0][0] not in start_lane:
                    start_lane.append(roadLink_lane_pair[ri][0][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = lane_pair




    return lane_phase_info_dict,intersection_road