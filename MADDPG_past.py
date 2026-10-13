import tensorflow as tf
#import tensorflow.contrib as tc
import numpy as np
import pandas as pd
import tensorflow.contrib as tc
import random

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
cross_state_shape= lane_in_eachcross*state_num  #24
core_each_memory_store=72
corner_each_memory_store=64
edge_each_memory_store=68




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
        self.epsilon = 0.9
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # 定义一个全局步数，方便恢复模型
        self.global_step = tf.Variable(0, trainable=False)
        self.pointer = 0
        self.phase_pointer = 0

        self.action_store = []
        if self.name in corner:
            self.memory = np.zeros((MEMORY_CAPACITY, corner_each_memory_store),
                                   dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        if self.name in core:
            self.memory = np.zeros((MEMORY_CAPACITY, core_each_memory_store),
                                   dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        if self.name in edge:
            self.memory = np.zeros((MEMORY_CAPACITY, edge_each_memory_store),
                                   dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
        self.n_a_r = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)













        try:
            self.image_summary = tf.image_summary
            self.scalar_summary = tf.scalar_summary
            self.histogram_summary = tf.histogram_summary
            self.merge_summary = tf.merge_summary
            self.SummaryWriter = tf.train.SummaryWriter
        except:
            self.image_summary = tf.summary.image
            self.scalar_summary = tf.summary.scalar
            self.histogram_summary = tf.summary.histogram
            self.merge_summary = tf.summary.merge
            self.SummaryWriter = tf.summary.FileWriter
        #模型存储位置
        self.MODEL_DIR='./model/{}'.format(self.name)

        self._build_net()
        self.merged = tf.summary.merge_all()

        if self.output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.train_writer = tf.summary.FileWriter("./{name}/logs/{g}".format(name=self.name,g=self.sess.graph))
        else:
            self.train_writer= None

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)




    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.reward= tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)  # reward
        self.state_input= tf.placeholder(shape=[None, 4, 6], dtype=tf.float32)  # 每个agent的state
        self.state_input_= tf.placeholder(shape=[None, 4, 6], dtype=tf.float32)  # 每个agent的state_

        self.action_input  = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)  # 每一个agent输出一个action
        # 角智能体
        if self.name in corner:
            self.other_action_input= tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)
        # 核心智能体
        if self.name in core:
            self.other_action_input= tf.placeholder(shape=[None, 4, 4], dtype=tf.float32)
        # 边智能体
        if self.name in edge:
            self.other_action_input= tf.placeholder(shape=[None, 4, 3], dtype=tf.float32)



        with tf.variable_scope('Actor'):
            self.a = self.actor_network(self.name+'actor',self.state_input,  trainable=True,scope='eval')
            a_ = self.actor_network(self.name+'actor_target',self.state_input_,  trainable=False,scope='target')
        self.action=tf.concat([self.a,self.other_action_input],axis=2)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.critic_output= self.critic_network(self.name+'critic',self.state_input, self.action, scope='eval', trainable=True)
            self.q_ = self.critic_network(self.name+'critic_target',self.state_input_, self.action,  scope='target', trainable=False)



        # '''--------------------------actor critic更新 -----------------------------'''
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        # self.replace_target_op = [tf.assign(a_t, c_t, a_e, c_e) for a_t, c_t, a_e, c_e in
        #                           zip(self.at_params, self.ct_params,self. ae_params, self.ce_params)]
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]


        #critic目标 ，最小化q_target和q值的差距
        #self.target_Q = tf.placeholder(shape=[None,4,1], dtype=tf.float32)

        with tf.variable_scope('loss'):
            q_target = self.reward + self.GAMMA * self.q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.critic_output)
            # if self.name in corner:
            #     self.td_error_1 = tf.reduce_mean(tf.squared_difference(self.q_, self.critic_output))
            #     self.actor_loss_1 = -tf.reduce_mean(self.critic_output+self.reward)
            # if self.name in core:
            #     self.td_error_2 = tf.reduce_mean(tf.squared_difference(self.q_, self.critic_output))
            #     self.actor_loss_2 = -tf.reduce_mean(self.critic_output+self.reward)
            # if self.name in edge:

                # self.td_error = tf.reduce_mean(tf.squared_difference(self.q_, self.critic_output))
            self.actor_loss = -tf.reduce_mean(self.critic_output-self.reward)

        with tf.variable_scope('train'):
            # if self.name in corner:
            #     self.actor_train = tf.train.AdamOptimizer(3e-10).minimize(self.actor_loss, var_list=self.ae_params,
            #                                                              global_step=self.global_step)
            #     self.critic_train = tf.train.AdamOptimizer(3e-10).minimize(self.td_error, var_list=self.ce_params,global_step=self.global_step)
            # if self.name in core:
            #     self.actor_train = tf.train.AdamOptimizer(3e-5).minimize(self.actor_loss, var_list=self.ae_params,
            #                                                              global_step=self.global_step)
            #     self.critic_train = tf.train.AdamOptimizer(3e-5).minimize(self.td_error, var_list=self.ce_params,
            #                                                               global_step=self.global_step)
            # if self.name in edge:
                self.actor_train = tf.train.AdamOptimizer(3e-5).minimize(self.actor_loss, var_list=self.ae_params,
                                                                         global_step=self.global_step)
                self.critic_train = tf.train.AdamOptimizer(3e-5).minimize(self.td_error, var_list=self.ce_params,
                                                                          global_step=self.global_step)

        #self.sess.run(tf.global_variables_initializer())
        #初始化图
        # tf.summary.scalar('c_loss_1',self.td_error_1)
        # tf.summary.scalar('a_loss_1', self.actor_loss_1)
        #
        # tf.summary.scalar('c_loss_2', self.td_error_2)
        # tf.summary.scalar('a_loss_2', self.actor_loss_2)

        tf.summary.scalar('c_loss_3', self.td_error)
        tf.summary.scalar('a_loss_3', self.actor_loss)








    def actor_network(self,name,state,trainable,scope):


        with tf.variable_scope(scope):

            state = tf.layers.dense(state, 104, name=name+'l1')
            # if self.layer_norm:
            #     state = tc.layers.layer_norm(state,center=True,scale=True,reuse=True)

            state = tf.nn.leaky_relu(state, name=name+'l2')


            # action = tf.layers.dense(state, 1, activation=tf.nn.sigmoid, name='action', trainable=trainable)
            action = tf.layers.dense(state, 1,
                                     activation=tf.nn.tanh,
                                kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                # use_bias=True,  #2020031622 跑，没改变没影响
                                trainable=trainable,
                                     name=name+'l3')
            # action = np.argmax(action)



            return action  #4*1

    def critic_network(self,cname,state,action_input,trainable,scope):
        #state = state_input
        #with tf.variable_scope(cname) as scope:
        with tf.variable_scope(scope):
            action=action_input     #24*2+4*3
            state_action=tf.concat([state,action],axis=2)
            critic_input_1=tf.layers.dense(state_action,360,name=cname+'l1')
            # if self.layer_norm:
            #     critic_input_1= tc.layers.layer_norm(critic_input_1,center=True,scale=True)

            # critic_input_2 = tf.layers.dense(critic_input_1, 72)
            #l1=tf.nn.sigmoid(critic_input_1)
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





    def train_actor(self, state,action, other_action,reward,sess):

        # sess.run(self.actor_train, {self.state_input: state, self.action_input:action,self.other_action_input: other_action,self.reward:reward})
    # #state：agent的state
        sess.run(self.actor_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,self.reward:reward})

    def train_critic(self, state, action,other_action, target, sess):
            sess.run(self.critic_train,
                  {self.state_input:state, self.action_input: action,self.other_action_input:other_action,
                   self.q_: target})


    def action_phase(self, state):
        action_probability = self.sess.run(self.a, {self.state_input: state})

        if random.random() < self.epsilon:
            # a=action_probability[0
            action=np.random.choice(4)
            # print('ram',action)
        else:
            action=np.argmax(action_probability)

        return action,action_probability

    def Q(self, state, action,other_action, sess):
         return sess.run(self.critic_output,
                         {self.state_input: state, self.action_input: action,self.other_action_input:other_action})

    def final_action(self,action_probability):

        return np.random.choice(action_probability[0].shape[1], size=1)

    def choice(self,action_probability):
        action=np.argmax(action_probability)
        # for i in range(4):
            # if i!=action :
            #     action_probability[0][i]=0

        return action, action_probability








    def phase(self,action,current_time):

        cur_a = action
        time=11

        if current_time == time:
            past_a=[2.]
        else:
            past_a=self.action_store[self.phase_pointer-1]



        if cur_a == past_a:
            self.state['change']=False
        else:
            self.state['change'] = True

        self.state['phase'] =cur_a
        self.state['past_phase']=past_a

        self.phase_pointer +=1


    def store_transition(self,each_past_now_r, each_agent_action,other_agent_action,type):  # each agent的 s ，s_ ，r ，a

        if type == 'corner':
            all_n = 8  # 4*2
        if type == 'core':
            all_n=16

        if type == 'edge':
            all_n=12

        for i in range(len(each_past_now_r)):
            a__ =other_agent_action[i]
            s_s__r=each_past_now_r[i]
            a=each_agent_action[i]

            a_p = [np.reshape(a[0], (1, 4))][0]
            action_p=a_p[0]

            a__p = np.reshape(a__, (1, all_n))
            other_agent_action_p = a__p[0]

            transition = np.hstack(
                (s_s__r, action_p, other_agent_action_p))
            index = self.pointer % MEMORY_CAPACITY
            self.memory[index] = transition
            self.pointer += 1
            # '''-----------------写入memory文件'''
            # if self.pointer >= MEMORY_CAPACITY:
            #     if self.name in corner:
            #         self.memory = np.zeros((MEMORY_CAPACITY, corner_each_memory_store),
            #                                dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
            #     if self.name in core:
            #         self.memory = np.zeros((MEMORY_CAPACITY, core_each_memory_store),
            #                                dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory
            #     if self.name in edge:
            #         self.memory = np.zeros((MEMORY_CAPACITY, edge_each_memory_store),
            #                                dtype=np.float32)  # s, s_,a_,r  每个agent有自己的memory







    def learn(self,epoch,type):


        self.sess.run(self.soft_replace)
        n_bt=15

        end=self.pointer
        start=end-1000

        #self.sess.run(tf.global_variables_initializer())
        #indices =np.random.randint(end,n_bt)
        indices = np.random.choice(range(start,end),size=n_bt)%MEMORY_CAPACITY
        #indices=range(start,end)
        # past_s=self.memory[:,0:32]
        # now_s=self.memory[:,32:64]
        # reward=self.memory[:,64]
        # action=self.memory[:,65:69]
        # other_action_input=self.memory[:,69:]
        # bt = self.memory[indices, :]
        bt = self.memory[indices, :]
        # past_s=self.memory[:,0:cross_state_shape]
        # now_s=self.memory[:,cross_state_shape:cross_state_shape*2]
        # reward=self.memory[:,cross_state_shape*2:cross_state_shape*2+4]
        # action=self.memory[:,cross_state_shape*2+4:(cross_state_shape*2)+8]
        # other_action_input=self.memory[:,(cross_state_shape*2)+8:]
        #
        past_s=bt[:,0:cross_state_shape]
        now_s=bt[:,cross_state_shape:cross_state_shape*2]
        reward=bt[:,cross_state_shape*2:cross_state_shape*2+4]
        action=bt[:,cross_state_shape*2+4:(cross_state_shape*2)+8]
        other_action_input=bt[:,(cross_state_shape*2)+8:]

        final_reward = reward
        final_action = action
        final_other_action = other_action_input
        #final_other_reward=other_reward_input
        past_s=np.array(past_s).reshape(n_bt,phase_num,state_num*2)
        now_s=np.array(now_s).reshape(n_bt,phase_num,state_num*2)
        reward=np.array(final_reward).reshape(n_bt,4,1)
        action=np.array(final_action).reshape(n_bt,4,1)
        if self.name in corner:
            other_action_input=np.array(final_other_action).reshape(n_bt,4,2)
        if self.name in core:
            other_action_input=np.array(final_other_action).reshape(n_bt,4,4)
        if self.name in edge:
            other_action_input=np.array(final_other_action).reshape(n_bt,4,3)




        self.sess.run(self.actor_train, {self.state_input: past_s, self.a:action,self.other_action_input:other_action_input,self.reward: reward,self.state_input_: now_s })

        self.sess.run(self.critic_train,
                      {self.state_input: past_s, self.a: action, self.other_action_input: other_action_input,
                       self.reward: reward, self.state_input_: now_s})


        c_l=self.sess.run(self.td_error,{self.state_input: past_s, self.a:action,self.other_action_input:other_action_input , self.reward: reward, self.state_input_: now_s})
        # # a_l=self.sess.run(self.actor_loss,{self.state_input:past_s,self.action_input:action,self.reward:reward})
        a_l = self.sess.run(self.actor_loss, {self.state_input: past_s, self.action_input: action, self.other_action_input: other_action_input,self.reward:reward,self.state_input_: now_s})

        # rs=self.sess.run(self.merged,feed_dict= {self.state_input: past_s, self.action_input: action, self.other_action_input: other_action_input,self.reward:reward,self.state_input_: now_s})
        # self.train_writer.add_summary(rs,epoch)
        self.save()

        return c_l,a_l


    def save(self):
        '''
        保存模型
        :return:
        '''
        # tf.train.Saver是用来保存训练结果的。
        # max_to_keep 用来设置最多保存多少个模型，默认是5
        # 如果保存的模型超过这个值，最旧的模型将被删除
        #print("new_global_step:",self.sess.run(self.global_step))

        self.saver.save(self.sess,self.MODEL_DIR+'myModel',global_step=self.global_step,write_meta_graph=False)

    def reload(self):
        '''
        恢复模型
        :return:
        '''
        self.model = tf.train.get_checkpoint_state(self.MODEL_DIR)


        if self.model and self.model.model_checkpoint_path:
            self.saver.restore(self.sess, self.model.model_checkpoint_path)
            # 读取当前已经训练的次数global_step,若不想从断点处开始训练，注释下面几行即可
            print("old_global_step:", self.sess.run(self.global_step))
            self.step = self.sess.run(self.global_step)
            print('Continue from')
            print('              ---> Minibatch update : ', self.step)

        print('开始训练')
























