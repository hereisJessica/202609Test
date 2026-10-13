from env import Cityflow
import numpy as np
from utility import parse_arguments
from utility import parse_roadnet,Logger
from MPLight.MARDDPG_2 import MARDDPG
import time
from each_agent import each_agent as cross
import os,sys
from dqn_agent import DQNAgent
from datetime import datetime

rewardlist=[]
# env=Cityflow()
#
# env.get_state()

args = parse_arguments()
def get_all_agent_action(all_state, all_agent):
    '''

    :param all_state:
    :param all_agent:
    :return: all_action
    '''
    all_action = []
    all_action_p = []

    shuaijian=0.8

    for i in range(len(all_agent)):
        each_state=all_state[i]
        state = np.reshape(each_state, [1, 4, 2])
        a,a_p = all_agent[i].choose_action(state)
        all_action.append(a)
        all_action_p.append(a_p)
    return all_action,all_action_p


if __name__ == "__main__":

    current_time = datetime.now()
    START_TIME = current_time.strftime("%Y-%m-%d-%H-%M")
    start=datetime.now()

    log_name='./Logs/MPLight_'+START_TIME+'.log'
    sys.stdout = Logger(log_name, sys.stdout)


    # init_start_time = time.time()
    config={}

    '''环境路径 '''
    ab_path=os.path.abspath('..')

    prefix_intersections = str(args.road_net)
    PATH_TO_DATA=os.path.join(ab_path,prefix_intersections)
    config_path=os.path.join(PATH_TO_DATA,"con_3_4.json")
    road_path=os.path.join(PATH_TO_DATA,"roadnet_3_4.json")
    env = Cityflow(config_path)
    lane_phase_info, intersection_road = parse_roadnet(road_path)
    intersection_id = list(lane_phase_info.keys())

    phase_list =[]
    for i in range(len(intersection_road)):
        phase_list.append(0)


    config['state_size'] = len(phase_list) * 12 * 2
    state_size = config['state_size']  #
    config['action_size'] = len(phase_list)
    config['lane_phase_info']=lane_phase_info
    phase_list = []
    for i in range(len(intersection_road)):
        phase_list.append(0)

    '''
    all_agent
    '''
    all_agent=[]
    for i in range(len(intersection_id)):
        all_agent.append(DQNAgent(intersection_id[i],config,args))

    # agent = DQNAgent(config)

    # 每个路口按相位分类,初始化路网
    env.get_each_intersction_lane(intersection_road)
    a_cross=cross(intersection_id)
    all_cross=a_cross.each_cross()

    # init_end_time=time.time()
    # init_run_time=init_end_time-init_start_time

    # env.Save_Run_Time(init_start_time, init_end_time,init_run_time, road_net_n=args.scenario,label='_init')


    for e in range(args.episodes):
        # reset initially at each episode
        train_start_time = time.time()
        env.reset()

        current_time = env.eng.get_current_time()
        t = 0
        while t < args.num_step:



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
                all_action=(np.ones(len(all_agent))).tolist()
                # all_action = [1, 1, 1, 1, 1, 1, 1, 1, 1]
                all_action_p = np.ones((len(all_agent),4),dtype = np.int)
            else:
                all_action,all_action_p = get_all_agent_action(each_intersection_past_state, all_agent)

            for k in range(len(all_agent)):
                # all_agent[k].state = each_intersection_past_state[k]

                # action
                all_agent[k].action_store.extend([all_action[k]])
                # get_phase
                all_agent[k].phase(all_action[k], current_time)
                # all_cross[k].other_state = np.zeros((1, 8))

                # for i in range(len(all_agent)):
                #
                #     all_cross[k].other_state += each_intersection_past_state[i]
                #


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
            env.store_to_csv(all_count, all_wait, reward,all_time,args,road_net_n=args.scenario)
            '''
            STORE
            '''
            # for i in range(len(all_agent)):
            #     all_cross[i].other_state_ = np.zeros((1, 8))
            #     for j in range(len(all_agent)):
            #         if j != i:
            #             all_cross[i].other_state_ += each_intersection_next_state[j]


            for i in range(len(all_agent)):
                all_agent[i].remember(each_intersection_past_state[i], all_action_p[i], reward_p[i],
                                      each_intersection_next_state[i],)
                all_agent[i].epsilon = np.max([all_agent[i].epsilon - 1e-4, 0.03])

            print("episode: {}/{}, time: {}, acton: {}, reward: {}"
                  .format(e, args.episodes, t - 1, all_action, reward))
            """learn"""
            if t != 0:
                if (t % 100) == 0:
                    print("------------------learn--------------------")
                    for agent in all_agent:
                        agent.learn()
                        # agent.save()
            # train_end_time= time.time()
            # train_run_time= train_end_time-train_start_time

            # env.Save_Run_Time(train_start_time,train_end_time,train_run_time,road_net_n=args.scenario)




    for ag in all_agent:
        ag.save(START_TIME)

    print('save')

    localtime =time.asctime(time.localtime(time.time()))
    end=datetime.now()

    print("ENDTIME:{0} Total_time:{1}".format(localtime,start-end))
    sys.stdout.write("ENDTIME:{0} Total_time:{1}".format(localtime, end - start))

