import os
from env import Cityflow
import numpy as np
from utility import parse_arguments
from utility import parse_roadnet
from dqn_agent import DQNAgent
import pandas as pd
import time

rewardlist=[]
# env=Cityflow()
#
# env.get_state()

args = parse_arguments()
#roadnet = '{}/roadnet.json'.format(args.scenario)
roadnet = '3_3/roadnet.json'

def get_all_agent_action(all_state, all_agent,current_time):
    '''

    :param all_state:
    :param all_agent:
    :return: all_action
    '''
    all_action = []
    all_action_p = []

    shuaijian=0.8

    for i in range(9):
        each_state=all_state[i]

        # if i == 0:
        #     if (all_agent[1].past_action == 1) or (all_agent[1].past_action == 2) or (all_agent[1].past_action == 4):
        #         each_state += all_state[1]*shuaijian
        #     if (all_agent[4].past_action == 1) or (all_agent[4].past_action == 2) or (all_agent[4].past_action == 3):
        #         each_state += all_state[4]*shuaijian
        # if i == 1:
        #     if (all_agent[0].past_action == 1) or (all_agent[0].past_action == 2) or (all_agent[0].past_action == 4):
        #         each_state += all_state[0]*shuaijian
        #     if (all_agent[5].past_action == 1) or (all_agent[5].past_action == 2) or (all_agent[5].past_action == 3):
        #         each_state += all_state[5]*shuaijian
        #     if (all_agent[0].past_action == 1) or (all_agent[0].past_action == 2) or (all_agent[0].past_action == 4):
        #         each_state += all_state[0] * shuaijian
        # if i == 2:
        #     if (all_agent[1].past_action == 1) or (all_agent[1].past_action == 2) or (all_agent[1].past_action == 4):
        #         each_state += all_state[1]*shuaijian
        #     if (all_agent[3].past_action == 1) or (all_agent[3].past_action == 2) or (all_agent[3].past_action == 4):
        #         each_state += all_state[3] * shuaijian
        #     if (all_agent[6].past_action == 1) or (all_agent[6].past_action == 2) or (all_agent[6].past_action == 3):
        #         each_state += all_state[6]*shuaijian
        # if i == 3:
        #     if (all_agent[2].past_action == 1) or (all_agent[2].past_action == 2) or (all_agent[2].past_action == 4):
        #         each_state += all_state[2]*shuaijian
        #     if (all_agent[7].past_action == 1) or (all_agent[7].past_action == 2) or (all_agent[7].past_action == 3):
        #         each_state += all_state[7]*shuaijian
        # if i == 4:
        #     if (all_agent[0].past_action == 1) or (all_agent[0].past_action == 2) or (all_agent[0].past_action == 3):
        #         each_state += all_state[0]*shuaijian
        #     if (all_agent[8].past_action == 1) or (all_agent[8].past_action == 2) or (all_agent[8].past_action == 3):
        #         each_state += all_state[8]*shuaijian
        #     if (all_agent[5].past_action == 1) or (all_agent[5].past_action == 2) or (all_agent[5].past_action == 4):
        #         each_state += all_state[5] * shuaijian
        # if i == 5:
        #     if (all_agent[1].past_action == 1) or (all_agent[1].past_action == 2) or (all_agent[1].past_action == 3):
        #         each_state += all_state[1]*shuaijian
        #     if (all_agent[9].past_action == 1) or (all_agent[9].past_action == 2) or (all_agent[9].past_action == 3):
        #         each_state += all_state[9]*shuaijian
        #     if (all_agent[4].past_action == 1) or (all_agent[4].past_action == 2) or (all_agent[4].past_action == 4):
        #         each_state += all_state[4] * shuaijian
        #     if (all_agent[6].past_action == 1) or (all_agent[6].past_action == 2) or (all_agent[6].past_action == 4):
        #         each_state += all_state[6] * shuaijian
        # if i == 6:
        #     if (all_agent[2].past_action == 1) or (all_agent[2].past_action == 2) or (all_agent[2].past_action == 3):
        #         each_state += all_state[2]*shuaijian
        #     if (all_agent[10].past_action == 1) or (all_agent[10].past_action == 2) or (all_agent[10].past_action == 3):
        #         each_state += all_state[10]*shuaijian
        #     if (all_agent[5].past_action == 1) or (all_agent[5].past_action == 2) or (all_agent[5].past_action == 4):
        #         each_state += all_state[5] * shuaijian
        #     if (all_agent[7].past_action == 1) or (all_agent[7].past_action == 2) or (all_agent[7].past_action == 4):
        #         each_state += all_state[7] * shuaijian
        # if i == 7:
        #     if (all_agent[3].past_action == 1) or (all_agent[3].past_action == 2) or (all_agent[3].past_action == 3):
        #         each_state += all_state[3]*shuaijian
        #     if (all_agent[11].past_action == 1) or (all_agent[11].past_action == 2) or (all_agent[11].past_action == 3):
        #         each_state += all_state[11]*shuaijian
        #     if (all_agent[6].past_action == 1) or (all_agent[6].past_action == 2) or (all_agent[6].past_action == 4):
        #         each_state += all_state[6] * shuaijian
        # if i == 8:
        #     if (all_agent[9].past_action == 1) or (all_agent[9].past_action == 2) or (all_agent[9].past_action == 4):
        #         each_state += all_state[9]*shuaijian
        #     if (all_agent[4].past_action == 1) or (all_agent[4].past_action == 2) or (all_agent[4].past_action == 3):
        #         each_state += all_state[4]*shuaijian
        # if i == 9:
        #     if (all_agent[8].past_action == 1) or (all_agent[8].past_action == 2) or (all_agent[8].past_action == 4):
        #         each_state += all_state[8]*shuaijian
        #     if (all_agent[10].past_action == 1) or (all_agent[10].past_action == 2) or (all_agent[10].past_action == 4):
        #         each_state += all_state[10] * shuaijian
        #     if (all_agent[5].past_action == 1) or (all_agent[5].past_action == 2) or (all_agent[5].past_action == 3):
        #         each_state += all_state[5]*shuaijian
        # if i == 10:
        #     if (all_agent[9].past_action == 1) or (all_agent[9].past_action == 2) or (all_agent[9].past_action == 4):
        #         each_state += all_state[9]*shuaijian
        #     if (all_agent[11].past_action == 1) or (all_agent[11].past_action == 2) or (all_agent[11].past_action == 4):
        #         each_state += all_state[11] * shuaijian
        #     if (all_agent[6].past_action == 1) or (all_agent[6].past_action == 2) or (all_agent[6].past_action == 3):
        #         each_state += all_state[6]*shuaijian
        # if i == 11:
        #     if (all_agent[10].past_action == 1) or (all_agent[10].past_action == 2) or (all_agent[10].past_action == 4):
        #         each_state += all_state[10]*shuaijian
        #     if (all_agent[7].past_action == 1) or (all_agent[7].past_action == 2) or (all_agent[7].past_action == 3):
        #         each_state += all_state[7]*shuaijian

        a,a_p = all_agent[i].choose_action(t,each_state,all_state,i)
        all_action.append(a)
        all_action_p.append(a_p)
    return all_action,all_action_p


if __name__ == "__main__":
    ## configuration for both environment and agent
    start_time=time.time()
    lane_phase_info, _ = parse_roadnet(roadnet)
    config = {
        'scenario': args.scenario,  # 场景
        'data': 'data/{}'.format(args.scenario),  # 据
        'roadnet': roadnet,  # 路网
        'flow': 'data/{}/flow.json'.format(args.scenario),  # 车流
        # 'replay_data_path': 'data/frontend/web',
        'num_step': args.num_step,  # 运行步数
        'lane_phase_info': lane_phase_info  # get lane and phase mapping by parsing the roadnet
        # 'initial_phase':
    }
    # all intersetion_id
    _, intersection_road = parse_roadnet(roadnet)
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
    phase_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    config['state_size'] = len(phase_list) * 12 * 2
    config['action_size'] = len(phase_list)

    # add visible gpu if necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # config_path= '/home/dl/PycharmProjects/sumo-cityflow/3_3/test.json'# ubuntu
    # config_path='3_3/test.json'#win
    #config_path = r'3_3/test.json'
    config_path='3_3/test.json'
    env = Cityflow(config_path)

    '''
    all_agent
    '''

    cross1_dqn = DQNAgent(intersection_id[0], config)
    cross2_dqn = DQNAgent(intersection_id[1], config)
    cross3_dqn = DQNAgent(intersection_id[2], config)
    cross4_dqn = DQNAgent(intersection_id[3], config)
    cross5_dqn = DQNAgent(intersection_id[4], config)
    cross6_dqn = DQNAgent(intersection_id[5], config)
    cross7_dqn = DQNAgent(intersection_id[6], config)
    cross8_dqn = DQNAgent(intersection_id[7], config)
    cross9_dqn = DQNAgent(intersection_id[8], config)
    # cross10_dqn = DQNAgent(intersection_id[9], config)
    # cross11_dqn = DQNAgent(intersection_id[10], config)
    # cross12_dqn = DQNAgent(intersection_id[11], config)
    # cross13_dqn = DQNAgent(intersection_id[12], config)
    # cross14_dqn = DQNAgent(intersection_id[13], config)
    # cross15_dqn = DQNAgent(intersection_id[14], config)
    # cross16_dqn = DQNAgent(intersection_id[15], config)

    # agent = DQNAgent(config)

    # 每个路口按相位分类,初始化路网
    env.get_each_intersction_lane(intersection_road)

    # some parameters in dqn
    batch_size = 32
    EPISODES = 120
    learning_start = 300
    update_model_freq = 300
    update_target_model_freq = 1500
    num_step = config['num_step']#3600
    state_size = config['state_size']#216
    all_agent = [cross1_dqn, cross2_dqn, cross3_dqn, cross4_dqn, cross5_dqn, cross6_dqn, cross7_dqn,
                 cross8_dqn, cross9_dqn]


    for e in range(EPISODES+1):
        # reset initially at each episode
        env.reset()

        current_time = env.eng.get_current_time()
        t = 0
        while t < num_step:

            '''
            1.获取all_state
            2.每个路口的状态 
             每个口 4*3=12个车道，每个车道有两个状态值（number，waiting time）12*2
            '''
            past_state = env.get_state(intersection_road)
            each_intersection_past_state,all_state_past,_,_ = env.get_each_intersection_state(past_state)  # 9*8 (72,) (9,) (9,)all_state_array,all_state,all_count,all_wait

            '''
            1.get_action  1*9
            2.判断是否是改变相位
            '''
            if t == 0:
                all_action = [1, 1, 1, 1, 1, 1, 1, 1, 1]
                all_action_p = np.ones((9,4),dtype = np.int)
            else:
                all_action,all_action_p = get_all_agent_action(each_intersection_past_state, all_agent,current_time)

            for k in range(len(all_agent)):
                all_agent[k].state = each_intersection_past_state[k]
                # action
                all_agent[k].action_store.extend([all_action[k]])
                # get_phase
                all_agent[k].phase(all_action[k], current_time)

            need_to_change_cross_dict = env.get_need_change_intersection(all_agent)
            '''
            执行动作
            1. yellow_run
            2. run
            '''
            t = env.take_action(need_to_change_cross_dict, t)
            '''
            更新状态
            '''
            # next_state = env.get_state(intersection_road)
            # each_intersection_next_state=env.get_each_intersection_state(intersection_road)
            next_state = env.get_state(intersection_road)
            each_intersection_next_state,all_state_next,all_count,all_wait = env.get_each_intersection_state(next_state)  # 9*24

            '''
            REWARD
            '''
            reward,reward_p,all_time = env.get_reward(intersection_road)#reward_p 是9*4的 reward是(9,)是对reward的纵向求和
            # rewardlist.append(reward)
            # reward_file=pd.DataFrame(rewardlist)
            # # if not os.path.exists('/reward'):
            # #     os.makedirs('reward')
            # reward_file.to_csv('./reward/reward.csv')
            env.store_to_csv(all_count, all_wait, reward,all_time)
            '''
            STORE
            '''
            for i in range(len(all_agent)):
                all_agent[i].remember(each_intersection_past_state[i], all_action_p[i], reward_p[i],
                                      each_intersection_next_state[i])
                all_agent[i].epsilon = np.max([all_agent[i].epsilon - 1e-4, 0.01])

            print("episode: {}/{}, time: {}, acton: {}, reward: {}"
                  .format(e, EPISODES, t - 1, all_action, reward))
            """learn"""
            if t != 0:
                if (t % 1800) == 0:
                    print("------------------learn--------------------")
                    loss1 = cross1_dqn.learn()
                    loss2 = cross2_dqn.learn()
                    loss3 = cross3_dqn.learn()
                    loss4 = cross4_dqn.learn()
                    loss5 = cross5_dqn.learn()
                    loss6 = cross6_dqn.learn()
                    loss7 = cross7_dqn.learn()
                    loss8 = cross8_dqn.learn()
                    loss9 = cross9_dqn.learn()

    # env.log()
    end_time=time.time()
    total_run_time=end_time-start_time
    print('total_rnning time',total_run_time)