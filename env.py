import cityflow
from utility import parse_arguments
import numpy as np
import pandas as pd
import time
import os
import csv

times = time.strftime('%Y_%m_%d-%H_%M', time.localtime(time.time()))

intersection_num=9
MIN_ACTION_TIME=20
yellow_time=5
each_reward_list=[]
each_count_list=[]
each_wait_list=[]
each_time_list=[]
class Cityflow():

    def __init__(self,path):

        self.eng = cityflow.Engine(path)
        self.t=time.strftime("%Y_%m_%d-%H_%M", time.localtime())

        # self.road_path=self.config["roadnet"]
        # self.num_step = self.config['num_step']
        # self.lane_phase_info = config['lane_phase_info']  # "intersection_1_1"
        # self.intersection_id = list(self.lane_phase_info.keys())[0]
        # self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        # self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        # self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]
        # self.current_phase = self.phase_list[0]
        # self.current_phase_time = 0
        # self.yellow_time = 5

        self.phase_log = []

    def step(self, next_phase):
        '''

        '''

        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1


        '''
        phase 
        '''

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        self.eng.next_step()
        self.phase_log.append(self.current_phase)

    def get_each_intersction_lane(self,intersection_road):
        #车道数量
        lane_num=3
        #
        direction=4

        #个路口的每个车道
        for intersection,road in intersection_road.items():

            phase={}
            for phase,v in road.items():
                a=v
                lane = {}
                each_lane_state={}
                each_lane = {}
                for road in range(len(v)):
                    if phase == 'PA':
                        min=1
                        max=3
                    if phase == 'PC':
                        min=1
                        max=3
                    if phase == 'PB':
                        min=0
                        max=1
                    if phase == 'PD':
                        min=0
                        max=1
                    for i in range (min,max):

                            # a=intersection[j]
                            a=str(v[road])+'_'+str(i)
                            each_lane_state['count']=0
                            each_lane_state['waitting']=0
                            each_lane[a]=each_lane_state

                intersection_road[intersection][phase]=each_lane


    def reset(self):
        self.eng.reset()
        self.phase_log = []




    def get_state(self,intersection_road):
        for intersection,roads in intersection_road.items():
            for road,lanes in roads.items():
                # state_phase['count']={self.eng.get_lane_vehicle_count()[lane] for lane in lanes }
                for lane,state in lanes.items():

                    a=self.eng.get_lane_vehicle_count()[lane]
                    # b=self.eng.get_lane_waiting_vehicle_count()[lane]*5
                    b = self.eng.get_lane_waiting_vehicle_count()[lane]
                    state={'count':a,'waitting':b}
                    intersection_road[intersection][road][lane]=state
                    # intersection_road[intersection][road][lane]['count']=self.eng.get_lane_vehicle_count()[lane]
                    # intersection_road[intersection][road][lane]['waitting'] = self.eng.get_lane_waiting_vehicle_count()[lane]*5
                    x=1
        state=intersection_road
        return state

    def get_each_intersection_state(self,state):

        each_count=[]
        each_wait=[]
        all_state_array=[]

        # for i in range (intersection_num):
        for inter, roads in state.items():
            each_state = []
            for road, lanes in roads.items():
                phase_state=[]
                a=[]
                for lane, s in lanes.items():


                    phase_state=np.concatenate((phase_state,np.array(list(state[inter][road][lane].values()))), axis=None)
                    # each_wait=np.concatenate((each_wait, state[inter][road][lane]['count']),axis=None)
                    # each_count=np.concatenate((each_count,state[inter][road][lane]['waitting']),axis=None)
                if road =='PA' or road =='PC':
                    phase_state=phase_state.reshape(4,2)
                if road =='PB' or road =='PD':
                    phase_state = phase_state.reshape(2, 2)
                phase_state=phase_state.sum(axis=0)
                all_state_array = np.concatenate((all_state_array, phase_state),
                                                 axis=None)
                A=phase_state[0]
                each_wait =np.concatenate((each_wait , phase_state[0]),
                                                 axis=None)
                each_count =np.concatenate((each_count , phase_state[1]),
                                                 axis=None)
        all_state=all_state_array
        all_state_array = all_state_array.reshape(len(state), 8)

        all_count=np.sum(each_count.reshape(len(state),4),axis=1)
        all_wait=np.sum(each_wait.reshape(len(state),4),axis=1)


        return all_state_array,all_state,all_count,all_wait

    def get_need_change_intersection(self,all_agent):
        need_to_change_cross_dict = {}
        for i in range(len(all_agent)):
            if all_agent[i].need_to_change:
                need_to_change_cross_dict[all_agent[i].name]=all_agent[i].phase_number
        return need_to_change_cross_dict



    def take_action(self,need_to_change_cross_dict,t):
        '''

        :param need_to_change_cross_dict:
        :return:
        '''


        for i in range(MIN_ACTION_TIME):
            use = False if i == 0 else True
            if len(need_to_change_cross_dict)==0:
                self.eng.next_step()
            else:
                self.phase_run(use,need_to_change_cross_dict)
                # print('111')
            t+=1
        return t


    def phase_run(self,use,need_to_change_cross_dict):
        '''
        k:intersection
        p:phase
        USE:是否执行过黄灯
        '''
        if use == False:
            for k, p in need_to_change_cross_dict.items():
                self.eng.set_tl_phase(k, int(p))
            self.eng.next_step()
        else:
            self.eng.next_step()




    def get_reward(self,intersection_road):

        all_reward=[]

        for inter, roads in intersection_road.items():
            each_state = []
            for road, lanes in roads.items():
                phase_reward=[]
                for lane, s in lanes.items():
                    lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()[lane]
                    reward = -1 * lane_waiting_vehicle_count
                    phase_reward=np.concatenate((phase_reward,reward),axis=None)
                phase_reward=np.sum(phase_reward)
                all_reward=np.concatenate((all_reward,phase_reward),axis=None)
        all_reward_p = all_reward.reshape(len(intersection_road), 4)
        all_reward=np.sum(all_reward_p,axis=1) #每一行相加
        t_time=self.eng.get_average_travel_time()
        return all_reward,all_reward_p,t_time


    def store_to_csv(self,all_count,all_wait,all_reward,all_time,args,label='train_',road_net_n=None):

        '''

        :param state:
        :param all_reward: list each_agent reward
        :return:
        '''
        '''
        1.store reward
        '''

        File_Path ='reward/{0}_{1}_{2}_{3}_{4}/'.format(args.alg,args.scenario,label,args.test,self.t)
        # File_Path = 'reward/{0}/'.format(args.scenario)
        if not os.path.exists(File_Path):
            os.makedirs(File_Path)

        '''
        1.store reward
        '''

        reward_file_name = "./" + File_Path + 'each_reward_file.csv'
        count_file_name = "./" + File_Path + 'each_count_file.csv'
        time_file_name =  "./" + File_Path + 'each_time_file.csv'
        wait_file_name = "./" + File_Path + 'each_wait_file.csv'

        # reward_data = np.array(when).reshape(1,-1)
        with open (reward_file_name, mode = 'a' ) as r_file:
            reward_writer = csv.writer(r_file)
            reward_writer.writerow(all_reward)
        with open (count_file_name, mode = 'a' ) as c_file:
            count_writer = csv.writer(c_file)
            count_writer.writerow(all_count)
        with open (time_file_name, mode = 'a' ) as t_file:
            time_writer = csv.writer(t_file)
            wait_time = np.array(all_time).reshape(1,-1)
            time_writer.writerow(wait_time)
        with open (wait_file_name, mode = 'a' ) as w_file:
            wait_writer = csv.writer(w_file)
            wait_writer.writerow(all_wait)



    def Save_Run_Time(self,start_time,end_time,run_time,road_net_n=None,label='train_'):

        dir=os.path.abspath('.')
        reward_dir=os.path.join(dir, 'MPLight_3_4_REWARD')
        time_dir = reward_dir+'/MPLight_reward_' + road_net_n + str(label) + '_' + times
        if not os.path.exists( reward_dir):
            os.makedirs( reward_dir)
        if not os.path.exists(time_dir):
            os.makedirs(time_dir)

        if label == 'train_':
            run_time_dir=os.path.join(time_dir, 'each_run_time_'+times)
            if os.path.exists(run_time_dir):
                with open(run_time_dir, 'a') as myfile:
                    myfile.write('{0},{1},{2}\n'.format(start_time, end_time, run_time))
            if os.path.exists(run_time_dir) == False:
                with open(run_time_dir,'w+') as myfile:
                    myfile.write('{0},{1},{2}\n'.format(start_time, end_time,run_time))
        else:
            #因爲init time 只有一次所以 w+即可
            with open(time_dir,'w+') as myfile:
                myfile.write('{0},{1},{2}\n'.format(start_time, end_time,run_time))

