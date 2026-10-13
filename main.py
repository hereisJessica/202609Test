from env import Cityflow
import numpy as np
from utility import parse_arguments
from utility import parse_roadnet
from dqn_agent import DQNAgent
from each_agent import each_agent
from MADDPG import MADDPG
import pandas as pd
from all_critic import all_critic
import tensorflow as tf
# env=Cityflow()
#
# env.get_state()

args = parse_arguments()
roadnet = '{}/roadnet.json'.format(args.scenario)

length_road=3
def phase(self, action, current_time):
    time = 0
    need_change_interstion = []
    if current_time == time:
        past_p = 0
    else:
        past_p = self.action_store[self.phase_pointer - 1]

    if action == past_p:
        self.need_to_change = False
    else:
        self.need_to_change = True
    self.phase_number = action
    self.pointer += 1


def get_all_agent_action(all_state, all_agent,current_time):
    '''

    :param all_state:
    :param all_agent:
    :return: all_action
    '''
    all_action=[]
    all_action_p=[]

    i = 0
    for agent in all_agent:
        if agent in corner_A:
            if i ==0:
                a=1
            # print('corner',i)
            corner_action,corner_action_p = corner_ddpg.choose_action(state=np.reshape(all_state[i], [1,4,2]),other=np.reshape(agent.other_state, [1,4, 2]))
            i+=1
            # print(current_time)
            phase(agent, corner_action, current_time=current_time)
            all_action.append(corner_action)
            all_action_p.append(corner_action_p)

        if agent in core_A:
            core_action,core_action_p = core_ddpg.choose_action(state=np.reshape(all_state[i], [1,4, 2]),other=np.reshape(agent.other_state, [1,4, 2]))
            # print('core',i)
            i += 1
            phase(agent, core_action, current_time=current_time)
            all_action.append(core_action)
            all_action_p.append(core_action_p)

        if agent in edge_A:

            edge_action,edge_action_p = edge_ddpg.choose_action(state=np.reshape(all_state[i], [1,4, 2]),other=np.reshape(agent.other_state, [1,4, 2]))
            # print('edge',i)
            i += 1
            phase(agent, edge_action, current_time=current_time)
            all_action.append(edge_action)
            all_action_p.append(edge_action_p)


    return all_action,all_action_p

def get_all_agent_s_r_a(all_cross_9,past_state,next_state,r,action):

    '''

    :param all_cross_9:  所有的路口
    :param past_state: s
    :param next_state: s_
    :param r: reward
    :return:

    1.获取到每个路口的s，s_, reward
    2.存储到对应智能体
    '''
    for i in range(9):
        all_cross_9[i].state=past_state[i]
        all_cross_9[i].state_next=next_state[i]
        all_cross_9[i].reward=r[i]
        all_cross_9[i].action=action[i]



    for agent in all_cross_9:
        if agent in corner_A:
            all_agent.corner.s.append(agent.state)
            all_agent.corner.s_.append(agent.state_next)
            all_agent.corner.r.append(agent.reward)
            all_agent.corner.a.append(agent.action)
            all_agent.corner.other_state.append(agent.other_state)
            all_agent.corner.other_state_.append(agent.other_state_)


        if agent in core_A:
            all_agent.core.s.append(agent.state)
            all_agent.core.s_.append(agent.state_next)
            all_agent.core.r.append(agent.reward)
            all_agent.core.a.append(agent.action)
            all_agent.core.other_state.append(agent.other_state)
            all_agent.core.other_state_.append(agent.other_state_)

        if agent in edge_A:
            all_agent.edge.s.append(agent.state)
            all_agent.edge.s_.append(agent.state_next)
            all_agent.edge.r.append(agent.reward)
            all_agent.edge.a.append(agent.action)
            all_agent.edge.other_state.append(agent.other_state)
            all_agent.edge.other_state_.append(agent.other_state_)


def get_other_state(all_cross_9,s):
    corner=[0,2,6,8]
    core=[4]
    edge=[1,3,5,7]

    for i in range(len(all_cross_9)):
        other = []
        if i == 0:
            other=np.concatenate((s[i+3],s[i+1]),axis=None)
        if i == 1:
            other=np.concatenate((s[i-1],s[i+3],s[i+1]),axis=None)
        if i == 2:
            other=np.concatenate((s[i-1],s[i+3]),axis=None)
        if i == 3:
            other=np.concatenate((s[i-3],s[i+3],s[i+1]),axis=None)
        if i == 4:
            other=np.concatenate((s[i-3],s[i-1],s[i+1],s[i+3]),axis=None)
        if i == 5:
            other = np.concatenate((s[i - 3],s[i - 1], s[i + 3]), axis=None)
        if i == 6:
            other=np.concatenate((s[i-3],s[i+1]),axis=None)
        if i == 7:
            other=np.concatenate((s[i-3],s[i-1],s[i+1]),axis=None)
        if i == 8:
            other = np.concatenate((s[i - 3], s[i - 1]), axis=None)
        if i in corner:
            other=np.sum(other.reshape(8,2),axis=1)
        if i in core:
            other = np.sum(other.reshape(8, 4), axis=1)
        if i in edge:
            other = np.sum(other.reshape(8,3), axis=1)
        all_cross_9[i].other_state=other

def get_other_state_(all_cross_9,s_):
    corner=[0,2,6,8]
    core=[4]
    edge=[1,3,5,7]

    for i in range(len(all_cross_9)):
        other_ = []
        if i == 0:
            other_=np.concatenate((s_[i+1],s_[i+3]),axis=None)
        if i == 1:
            other_=np.concatenate((s_[i-1],s_[i+3],s_[i+1]),axis=None)
        if i == 2:
            other_=np.concatenate((s_[i-1],s_[i+3]),axis=None)
        if i == 3:
            other_=np.concatenate((s_[i-3],s_[i+3],s_[i+1]),axis=None)
        if i == 4:
            other_ = np.concatenate((s_[i - 3], s_[i - 1], s_[i + 1], s_[i + 3]), axis=None)
        if i == 5:
            other_ = np.concatenate((s_[i - 3],s_[i - 1], s_[i + 3]), axis=None)
        if i == 6:
            other_=np.concatenate((s_[i-3],s_[i+1]),axis=None)
        if i == 7:
            other_=np.concatenate((s_[i-3],s_[i-1],s_[i+1]),axis=None)
        if i == 8:
            other_ = np.concatenate((s_[i - 3], s_[i - 1]), axis=None)
        if i in corner:
            other_=np.sum(other_.reshape(8,2),axis=1)
        if i in core:
            other_ = np.sum(other_.reshape(8, 4), axis=1)
        if i in edge:
            other_ = np.sum(other_.reshape(8,3), axis=1)

        all_cross_9[i].other_state_=other_


def store_c_loss(file_c_l, c_l, name):
    # a = [c_l, a_l]
    file_c_l.append(c_l)
    c_l_file = pd.DataFrame(file_c_l)
    file_name = './reward/{}'.format(name)
    c_l_file.to_csv(file_name)

if __name__ == "__main__":
    ## configuration for both environment and agent

    lane_phase_info,_=parse_roadnet(roadnet)
    config = {
        'scenario': args.scenario, # 场景
        'data': 'data/{}'.format(args.scenario), #据
        'roadnet': roadnet,#路网
        'flow': 'data/{}/flow.json'.format(args.scenario),#车流
        #'replay_data_path': 'data/frontend/web',
        'num_step': args.num_step, #运行步数
        'lane_phase_info': lane_phase_info  # get lane and phase mapping by parsing the roadnet
        # 'initial_phase':
    }
    # all intersetion_id
    _,intersection_road=parse_roadnet(roadnet)
    intersection_id = list(config['lane_phase_info'].keys())
    '''
    all intersection phase  
    设置初始phase
    st:
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    config['state_size'] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
    config['action_size'] = len(phase_list)
    
    e:
    9(intersection) 
    4(direction) * 3 (lane)
    attribute: num and waiting
    '''
    phase_list = [0,0,0,0,0,0,0,0,0]
    config['state_size'] = len(phase_list) *12 *2
    config['action_size'] = len(phase_list)

    # add visible gpu if necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # config_path= '/home/dl/PycharmProjects/sumo-cityflow/3_3/test.json'# ubuntu
    config_path='3_3/test.json'
    env = Cityflow(config_path,config)


    '''
    all_agent
    
    '''
    all_agent = each_agent(intersection_id)
    edge_A = [all_agent.cross2,all_agent.cross4,all_agent.cross6, all_agent.cross8]
    core_A = [all_agent.cross5]
    corner_A = [all_agent.cross1,all_agent.cross3, all_agent.cross7,all_agent.cross9]
    #
    corner_ddpg = MADDPG('corner',config)
    core_ddpg = MADDPG('core',config)
    edge_ddpg = MADDPG('edge',config)


    all_cross_9 = [all_agent.cross1, all_agent.cross2, all_agent.cross3, all_agent.cross4, all_agent.cross5,
                   all_agent.cross6, all_agent.cross7, all_agent.cross8, all_agent.cross9]


    #每个路口按相位分类,初始化路网
    env.get_each_intersction_lane(intersection_road)



    # some parameters
    batch_size = 32
    EPISODES = 100
    learning_start = 300
    update_model_freq = 300
    update_target_model_freq = 1500
    num_step = config['num_step']
    state_size = config['state_size']


    for e in range(EPISODES):
        # reset initially at each episode
        env.reset()

        current_time=env.eng.get_current_time()
        t = 1
        while t < num_step:


            '''
            1.获取all_state
            2.每个路口的状态 
             每个口 4*3=12个车道，每个车道有两个状态值（number，waiting time）12*2
            '''
            past_state = env.get_state(intersection_road)#是一个dict（有9个内容）
            each_intersection_past_state,all_state_past,_,_=env.get_each_intersection_state(past_state)  #9*24

            '''
            1.get_action  1*9
            2.判断是否是改变相位
            '''
            current_time = env.eng.get_current_time()
            get_other_state(all_cross_9,each_intersection_past_state)
            all_action,all_action_p=get_all_agent_action(each_intersection_past_state,all_cross_9,current_time)


            for k in range(len(all_cross_9)):

                all_cross_9[k].state = each_intersection_past_state[k]
                # action
                all_cross_9[k].action_store.extend([all_action[k]])
                #get_phase
                # all_agent[k].phase(all_action[k],current_time)

            need_to_change_cross_dict = env.get_need_change_intersection(all_cross_9)
            '''
            执行动作
            1. yellow_run
            2. run
            '''
            t=env.take_action(need_to_change_cross_dict,t)

            '''
            更新状态
            '''
            # next_state = env.get_state(intersection_road)
            # each_intersection_next_state=env.get_each_intersection_state(intersection_road)
            next_state = env.get_state(intersection_road)#返回有9个数据的字典
            each_intersection_next_state,all_state_next,all_count,all_wait=env.get_each_intersection_state(next_state)  #9*24

            '''
            REWARD
            '''
            reward,reward_p = env.get_reward(intersection_road)#reward 是9*4的 reward_p是(9,)是对reward的纵向求和

            '''
            存储
            '''
            env.store_to_csv(all_count,all_wait,reward)

            '''
            get_other_state_
            '''
            get_other_state_(all_cross_9,each_intersection_next_state)

            '''
            STORE
            '''
            get_all_agent_s_r_a(all_cross_9, each_intersection_past_state, each_intersection_next_state, reward_p,
                                all_action_p)

            '''
            每个智能体获取到了各自的actioh，state，state_,reward\
            
            有三个智能体
            
            '''




            corner_ddpg.store_transition(all_agent.corner.s,all_agent.corner.s_,all_agent.corner.r,all_agent.corner.a,all_agent.corner.other_state,all_agent.corner.other_state_,'corner')
            core_ddpg.store_transition(all_agent.core.s,all_agent.core.s_,all_agent.core.r,all_agent.core.a,all_agent.core.other_state,all_agent.core.other_state_,'core')
            edge_ddpg.store_transition(all_agent.edge.s,all_agent.edge.s_,all_agent.edge.r,all_agent.edge.a,all_agent.edge.other_state,all_agent.edge.other_state_,'edge')
            # critic.store_transition(all_state_past,all_action,reward,all_state_next)


            corner_ddpg.epsilon=np.max( [ corner_ddpg.epsilon - 1e-4,0.03])
            core_ddpg.epsilon = np.max([core_ddpg.epsilon - 1e-4, 0.03])
            edge_ddpg.epsilon = np.max([edge_ddpg.epsilon - 1e-4, 0.03])




            # for i in range(len(all_agent_9)):
            #     all_agent_9[i].remember(each_intersection_past_state[i],all_action[i],reward[i],each_intersection_next_state[i])
            #     all_agent_9[i].epsilon = np.max([all_agent[i].epsilon - 1e-4, 0.01])

            print("episode: {}/{}, time: {}, acton: {}, reward: {}"
                  .format(e, EPISODES, t-1, all_action, reward))

            '''
            learn
            '''

            '''
            '''

            if (t-1) % 1000 == 0:
                print('----------learn')
                corner_c_l = corner_ddpg.learn()
                core_c_l = core_ddpg.learn()
                edge_c_l = edge_ddpg.learn()

                store_c_loss(all_agent.corner.loss_store, corner_c_l, 'corner')
                store_c_loss(all_agent.core.loss_store, core_c_l, 'core')
                store_c_loss(all_agent.edge.loss_store, edge_c_l, 'edge')



















        '''
        
        '''


        # # last_action = phase_list[agent.choose_action(state)]
        # while t < num_step:
        #     action = phase_list[agent.choose_action(state)]
        #     if action == last_action:
        #         env.step(action)
        #     else:
        #         for _ in range(env.yellow_time):
        #             env.step(0)  # required yellow time
        #             t += 1
        #             flag = (t >= num_step)
        #             if flag:
        #                 break
        #         if flag:
        #             break
        #         env.step(action)
        #     last_action = action
        #     t += 1
        #     next_state = env.get_state()
        #     reward = env.get_reward()
        #     next_state = np.array(list(next_state['start_lane_vehicle_count'].values()) + [next_state['current_phase']])
        #     next_state = np.reshape(next_state, [1, state_size])
        #     agent.remember(state, action, reward, next_state)
        #     state = next_state
        #
        #     total_time = t + e * num_step
        #     if total_time > learning_start and total_time % update_model_freq == update_model_freq - 1:
        #         agent.replay()
        #     if total_time > learning_start and total_time % update_target_model_freq == update_target_model_freq -1:
        #         agent.update_target_network()
        #
        #     print("episode: {}/{}, time: {}, acton: {}, reward: {}"
        #           .format(e, EPISODES, t-1, action, reward))
        #
        # if e % 10 == 0:
        #     if not os.path.exists("model"):
        #         os.makedirs("model")
        #     agent.model.save("model/trafficLight-dqn-{}.h5".format(e))

    # log environment files
    env.log()


