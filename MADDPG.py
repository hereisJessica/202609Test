import tensorflow as tf
#import tensorflow.contrib as tc
import numpy as np
import pandas as pd
import tensorflow.contrib as tc
import random
from collections import deque

# edge=['agent2','agent4','agent6','agent8','agent2_target','agent4_target','agent6_target','agent8_target']
# core=['agent5','agent5_target']
# corner=['agent1','agent1_target','agent3','agent7','agent9','agent3_target','agent7_target','agent9_target']

corner=['corner','corner_target']
core=['core','core_target']
edge=['edge','edge_target']

cross_num=4
phase_num=4
lane_in_eachcross=8
state_num=3
MEMORY_CAPACITY=50000
file_a_q,file_c_l,=[],[]

GLOBAL_NET_SCOPE='Global_Net'
cross_state_shape=8  #24

core_each_memory_store=32     #s:8,s_8,r:1,a:1,s_o:8,s_o_：8
corner_each_memory_store=146   #s:24,s_24,r:1,a:1,s_o:24*2,s_o_：24*2
edge_each_memory_store=194    #s:24,s_24,r:1,a:1,s_o:24*3,s_o_：24*3

memory_store=40 #s:8,s_8,r:4,a:4,s_o:8,s_o_：8


#s,s_,action_p,r,other_action_p


TAU = 0.01      # soft replacement

class MADDPG():
    def __init__(self,name,nb_input=24,nb_action=4,layer_norm=True):


        self.GAMMA = 0.999
        self.layer_norm = layer_norm
        self.nb_action=nb_action
        self.nb_input=nb_input
        self.state=None
        self.output_graph = True
        self.name=name
        self.epsilon = 1
        self.action_size = 4
        self.lr=1e-3

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 定义一个全局步数，方便恢复模型
        self.global_step = tf.Variable(0, trainable=False)
        self.pointer = 0
        self.phase_pointer = 0

        self.s_dim=24
        self.a_dim=1
        self.r_dim=1


        self.a_grads=None

        self.action_store = []
        # self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim *2 + self.a_dim + self.r_dim), dtype=np.float32)
        self.memory = np.zeros((MEMORY_CAPACITY, memory_store),
                               dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory

        # if self.name in corner:
        #     self.other_dim=24*2
        #     self.memory = np.zeros((MEMORY_CAPACITY, corner_each_memory_store),
        #                            dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        # if self.name in core:
        #     self.other_dim = 24 * 4
        #     self.memory = np.zeros((MEMORY_CAPACITY, core_each_memory_store),
        #                            dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        # if self.name in edge:
        #     self.other_dim = 24 * 3
        #     self.memory = np.zeros((MEMORY_CAPACITY, edge_each_memory_store),
        #                            dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        # self.n_a_r = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)


        #模型存储位置
        self.MODEL_DIR='./model/{}'.format(self.name)

        self._build_net()
        self.merged = tf.summary.merge_all()
        #
        # if self.output_graph:
        #     # $ tensorboard --logdir=logs
        #     # tf.train.SummaryWriter soon be deprecated, use following
        #     self.train_writer = tf.summary.FileWriter("./{name}/logs/{g}".format(name=self.name,g=self.sess.graph))
        # else:
        #     self.train_writer= None

        self.sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()





    def _build_net(self):

        # ------------------ all inputs ------------------------
        self.reward= tf.placeholder(shape=[None, 4,1], dtype=tf.float32)  # reward
        self.state_input= tf.placeholder(shape=[None, 4,2], dtype=tf.float32)  # 每个agent的state
        self.state_input_= tf.placeholder(shape=[None, 4,2], dtype=tf.float32)  # 每个agent的state_

        self.action  = tf.placeholder(shape=[None, 4,1], dtype=tf.float32)  # 每一个agent输出一个action

        self.other_state = tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)
        self.other_state_ = tf.placeholder(shape=[None, 4,2], dtype=tf.float32)

        with tf.variable_scope(self.name+'Actor'):

            self.L3,self.a = self.actor_network(self.name+'actor',self.state_input, self.other_state,trainable=True,scope='eval')
            a_ = self.actor_network(self.name+'actor_target',self.state_input_, self.other_state_, trainable=False,scope='target')

        with tf.variable_scope(self.name+'Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.critic_output = self.critic_network(self.name + 'Critic', self.state_input,self.other_state, self.L3, scope='eval',
                                                     trainable=True)
            self.q_ = self.critic_network(self.name + 'critic_target', self.state_input_,self.other_state_, self.L3, scope='target',
                                          trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'Critic/target')


        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]




        # '''--------------------------actor 更新 -----------------------------'''
        with tf.variable_scope(self.name+'loss'):
            q_target = self.reward + self.GAMMA * self.q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.critic_output)
            # self.actor_loss = -tf.reduce_mean(self.critic_output - self.a)
            self.actor_loss = - tf.reduce_mean(self.critic_output+self.L3)
        with tf.variable_scope(self.name+'train'):
            self.actor_train = tf.train.AdamOptimizer(3e-5).minimize(self.actor_loss, var_list=self.ae_params)
            self.critic_train = tf.train.AdamOptimizer(3e-5).minimize(self.td_error, var_list=self.ce_params)





    def actor_network(self,name,state,other_state,trainable,scope):
        with  tf.variable_scope(scope):

            #
            # init_w_s_o = tf.random_normal_initializer(1e-6, 1e-5)
            # init_b_s_o = tf.constant_initializer(1e-6)
            #
            # init_w_s = tf.random_normal_initializer(1e-2, 1e-1)
            # init_b_s = tf.constant_initializer(1e-1)
            #
            #
            # n_l1 = 414
            # w1_s_o = tf.get_variable(name+'w1_s_o', [self.other_dim, n_l1], initializer=init_w_s_o)
            # b1_o = tf.get_variable(name+'b1_o', [1, n_l1], initializer=init_b_s_o)
            # s_other = tf.nn.relu(tf.matmul(other_state, w1_s_o) + b1_o)
            #
            # w1_s = tf.get_variable(name+'w1_s', [24, n_l1], initializer=init_w_s)
            # b1_s = tf.get_variable(name+'b1', [1, n_l1], initializer=init_b_s)
            # s = tf.nn.relu(tf.matmul(state, w1_s) + b1_s)
            #
            # input = tf.concat([s,s_other],axis=1)

            with tf.variable_scope(scope):
                L1 = tf.layers.dense(state, 52, name=name + 'l1')
                # if self.layer_norm:
                #     state = tc.layers.layer_norm(state,center=True,scale=True,reuse=True)

                L2 = tf.nn.leaky_relu(L1, name=name + 'l2')

                # action = tf.layers.dense(state, 1, activation=tf.nn.sigmoid, name='action', trainable=trainable)
                L3 = tf.layers.dense(L2, 1,
                                         activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_uniform_initializer(minval=5e-3, maxval=5e-2),
                                         bias_initializer=tf.random_uniform_initializer(minval=5e-3, maxval=5e-2),
                                         # use_bias=True,  #2020031622 跑，没改变没影响
                                         trainable=trainable,
                                         name=name + 'l3')
                action = tf.argmax(L3, 1)
                action = tf.to_float(action)  # 24*2+4*3
                action = tf.layers.flatten(action)

                return L3,action  # 4*1


    def critic_network(self,cname,state,other_state,action_input,trainable,scope):

        with tf.variable_scope(scope):
            action=action_input
            state_action=tf.concat([state,other_state,action],axis=2)
            critic_input_1=tf.layers.dense(state_action,360,name=cname+'l1')

            q = tf.layers.dense\
                (critic_input_1, 1,
                 activation=tf.nn.leaky_relu,
                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 trainable=trainable,
                 name=cname+'l2')
            #q= tf.layers.dense(critic_input_1, 1, activation=tf.nn.leaky_relu, trainable=trainable)  # Q(s,a)

            # q = tf.layers.dense(critic_input_1, 1, activation=tf.nn.softmax, trainable=trainable)  # Q(s,a)
            q= tf.nn.relu(q)
            return q

    def choose_action(self, state,other):

        P, act_values = self.sess.run((self.L3, self.a), {self.state_input: state, self.other_state: other})
        if np.random.rand() <= self.epsilon:
            # a=random.randrange(self.action_size)
            return random.randrange(1,5),P #输出1-4
        else:
            P,act_values = self.sess.run((self.L3,self.a), {self.state_input: state,self.other_state:other})
            # print('-----------unram',P)
        return (act_values[0].tolist())[0]+1,P







    def phase(self, action, current_time):
        time = 0
        need_change_interstion = []
        if current_time == time:
            past_p = 0
        else:
            past_p = self.action_store[self.phase_pointer - 1]

        if action == past_p:
            self.need_to_change=False
        else:
            self.need_to_change=True
        self.phase_number=action
        self.phase_pointer += 1


    def store_transition(self,s,s_,r,a,s_other,s_other_,type):  # each agent的 s ，s_ ，r ，a

        for i in range(len(a)):

            transition = np.concatenate((s[i],s_[i],r[i],a[i],s_other[i],s_other_[i]),axis=None)
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index, :] = transition
            self.pointer += 1







    def learn(self):

        self.sess.run(self.soft_replace)
        n_bt=15

        end=self.pointer
        start=end-n_bt

        #self.sess.run(tf.global_variables_initializer())
        #indices =np.random.randint(end,n_bt)
        indices = np.random.choice(range(0,end),size=n_bt)
        indices = indices % MEMORY_CAPACITY
        bt = self.memory[indices, :]

        past_s=bt[:,0:cross_state_shape]
        now_s=bt[:,cross_state_shape:cross_state_shape*2]
        reward=bt[:,cross_state_shape*2:cross_state_shape*2+4]
        action=bt[:,cross_state_shape*2+4:(cross_state_shape*2)+8]
        other_s=bt[:, -cross_state_shape*2:-cross_state_shape]
        other_s_=bt[:, -cross_state_shape:]

        past_s=np.array(past_s).reshape(n_bt,phase_num,2)
        now_s=np.array(now_s).reshape(n_bt,phase_num,2)
        reward=np.array(reward).reshape(n_bt,4,1)
        action=np.array(action).reshape(n_bt,4,1)
        other_s=np.array(other_s).reshape(n_bt,4,2)
        other_s_=np.array(other_s_).reshape(n_bt,4,2)

        # if self.name in corner:
        #     other_s = bt[:, (cross_state_shape * 2) + 2:(cross_state_shape * 4) + 2]
        #     other_s_ = bt[:, -cross_state_shape*2:]
        #     other_state_input=np.array(other_s).reshape(n_bt,cross_state_shape*2)
        #     other_state_input_ = np.array(other_s_).reshape(n_bt,cross_state_shape*2)
        # if self.name in core:
        #     other_s = bt[:, (cross_state_shape * 2) + 2:(cross_state_shape * 6) + 2]
        #     other_s_ = bt[:, -cross_state_shape*4:]
        #     other_state_input=np.array(other_s).reshape(n_bt,cross_state_shape*4)
        #     other_state_input_ = np.array(other_s_).reshape(n_bt, cross_state_shape*4)
        # if self.name in edge:
        #     other_s = bt[:, (cross_state_shape * 2) + 2:(cross_state_shape * 5) + 2]
        #     other_s_ = bt[:, -cross_state_shape * 3:]
        #     other_state_input = np.array(other_s).reshape(n_bt, cross_state_shape * 3)
        #     other_state_input_ = np.array(other_s_).reshape(n_bt, cross_state_shape * 3)

        self.sess.run(self.actor_train, {self.state_input:past_s, self.other_state:other_s,self.L3:action})

        self.sess.run(self.critic_train,
                      {self.state_input: past_s, self.L3: action, self.other_state:other_s,self.other_state_:other_s_,
                       self.reward: reward, self.state_input_: now_s})

        c_l = self.sess.run(self.td_error,
                            {self.state_input: past_s, self.L3: action, self.other_state:other_s,self.other_state_:other_s_,
                       self.reward: reward, self.state_input_: now_s})

        return c_l
























