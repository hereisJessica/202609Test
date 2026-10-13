import numpy as np

MEMORY_CAPACITY=50000
each_store=24#s=8 a=4 r=4 s'=8
state_shape=8


class each_agent(object):
    def __init__(self,intersection_id):
        self.intersection=intersection_id
        # self.all_cross=[]
        # self.cross1 = cross(self.intersection[0])
        # self.cross2 = cross(self.intersection[1])
        # self.cross3 = cross(self.intersection[2])
        # self.cross4 = cross(self.intersection[3])
        # self.cross5 = cross(self.intersection[4])
        # self.cross6 = cross(self.intersection[5])
        # self.cross7 = cross(self.intersection[6])
        # self.cross8 = cross(self.intersection[7])
        # self.cross9 = cross(self.intersection[8])
        # self.cross10 = cross(self.intersection[9])
        # self.cross11 = cross(self.intersection[10])
        # self.cross12 = cross(self.intersection[11])
        # self.cross13 = cross(self.intersection[12])
        # self.cross14 = cross(self.intersection[13])
        # self.cross15 = cross(self.intersection[14])
        # self.cross16 = cross(self.intersection[15])
        #
        # self.corner=corner()
        # self.edge=edge()
        # self.core=core()


        # self.time=s_agent.get_current_time()
        self.state_up=False
        self.pause=False
        self.begin=False
        self.pointer=0
        self.phase_num=None

        self.other_state = []
        self.other_state_=[]

    def each_cross(self):
        all_cross=[]
        for i in range(len(self.intersection)):
            all_cross.append(cross(self.intersection[i]))
        return all_cross

    def phase(self, action, current_time,cross):
        time = 0
        need_change_interstion = []
        if current_time == time:
            past_p = 0
        else:
            past_p = cross.action_store[cross.phase_pointer - 1]

        if action == past_p:
            cross.need_to_change=False
        else:
            cross.need_to_change=True
        cross.phase_number=action
        cross.phase_pointer += 1





class cross(object):
    def __init__(self,name):
        self.name=name
        self.done=False

        #放网络里用的state
        self.state=0

        self.state_dict=None

        self.state_next=None

        self.reward=None

        self.action=None

        #存储其他agent的actin
        self.other_action=[]
        #存储其他agent的rewardc1_o_r
        self.other_reward=None
        #存储标记
        self.store=False
        self.loss_store=[]
        self.train=False
        self.pointer=0
        self.action_store=[]
        self.other_state = []
        self.other_state_=[]
        self.phase_pointer = 0
        self.need_to_change=False
        self.memory=np.zeros((MEMORY_CAPACITY,each_store))



        #cross1_action_p,cross1_other_action_input,c1_o_r


class corner(object):
    def __init__(self):
        self.s=[]
        self.s_=[]
        self.a=[]
        self.r=[]
        self.loss_store=[]
        self.action_p=[]
        self.other_state = []
        self.other_state_=[]


class edge(object):
    def __init__(self):
        self.s=[]
        self.s_=[]
        self.a=[]
        self.r=[]
        self.loss_store=[]
        self.action_p=[]
        self.other_state = []
        self.other_state_=[]


class core(object):
    def __init__(self):
        self.s=[]
        self.s_=[]
        self.a=[]
        self.r=[]
        self.loss_store=[]
        self.action_p=[]
        self.other_state = []
        self.other_state_=[]


