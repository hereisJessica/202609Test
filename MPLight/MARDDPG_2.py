import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops.rnn import dynamic_rnn
import os

cross_num = 9
MEMORY_CAPACITY = 50000
memory_store = 40  # s:8,s_8,r:4,a:4,s_o:8,s_o_：8
TAU = 0.01
cross_state_shape = 8


def softmax(X):
    """
    softmax函数实现

    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度

    返回：
    softmax计算结果
    """
    assert (len(X.shape) == 2)
    row_max = np.max(X, axis=0).reshape(-1, 1)
    X -= row_max
    # X2 = np.power(100,[X])
    # sum = np.sum(X2)
    # for i in range(X2.shape[0]):
    #     for j in range(X2.shape[1]):
    #         X2[i][j] = X2[i][j] / sum
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=0, keepdims=True)
    # sum = np.sum(X, axis=0)[0]
    # if sum == 0:
    #     X = np.array([0.25, 0.25, 0.25, 0.25])
    #     X = X.reshape(4, 1)
    # else:
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             X[i][j] = X[i][j] / sum

    return s


class MARDDPG():
    def __init__(self, name, config, layer_norm=True):
        self.GAMMA = 0.999
        self.layer_norm = layer_norm

        self.name = name
        self.epsilon = 1
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.global_step = tf.Variable(0, trainable=False)
        self.s_dim = 8
        self.a_dim = 4
        self.r_dim = 4
        self.a_grads = None
        # self.phase=None
        self.action_store=[]
        self.batch_size = 16
        self.phase_pointer=0

        self.saver = tf.train.Saver()

        # memeory_pointer
        self.pointer = 0

        self.lstm_size = 8  # hidden_size
        self.min_history = 4
        self.num_lstm_layers = 1

        self.memory = np.zeros((MEMORY_CAPACITY, memory_store),
                               dtype=np.float32)
        self.MODEL_DIR = './model/{}'.format(self.name)

        self._build_net()
        self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()

    def _build_net(self):
        self.state = tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None, 4, 1], dtype=tf.float32)
        self.state_ = tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)

        self.other_state = tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)
        self.other_state_ = tf.placeholder(shape=[None, 4, 2], dtype=tf.float32)

        # create placeholder to fill in lstm state
        self.c_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        self.h_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        self.lstm_state_actor = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)

        self.c_state_critic_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        self.h_state_critic_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")
        self.c_state_critic_eval = tf.placeholder(tf.float32, [None, self.lstm_size], name="eval_c")
        self.h_state_critic_eval = tf.placeholder(tf.float32, [None, self.lstm_size], name="eval_h")
        self.lstm_state_critic_eval = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_critic_eval, self.h_state_critic_eval)
        self.lstm_state_critic_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_critic_target,
                                                                      self.h_state_critic_target)

        # inital zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((self.batch_size, self.lstm_size))
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))

        self.lstm_state_c, self.lstm_state_h = np.zeros((1, self.lstm_size)), np.zeros((1, self.lstm_size))

        # 创建actor_eval和actor_terget
        with tf.variable_scope(self.name + 'Actor'):
            self.L3 = self.actor_network(self.name + 'actor', self.state, self.lstm_state_actor,
                                         trainable=True, scope='eval')
            self.a_ = self.actor_network(self.name + 'actor_target', self.state_, self.lstm_state_actor,
                                         trainable=False, scope='target')
        # 创建critic_target和critic_eval
        with tf.variable_scope(self.name + 'Critic'):
            # 这时输入的action是4*1，改为了1*1
            self.critic_output = self.critic_network(self.name + 'critic', self.state, self.other_state,
                                                     self.L3, self.lstm_state_critic_eval, scope='eval',
                                                     trainable=True)
            self.q_ = self.critic_network(self.name + 'critic_target', self.state_, self.other_state_, self.L3,
                                          self.lstm_state_critic_target,
                                          scope='target',
                                          trainable=False)

            # 初始化参数
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Actor/target')
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Critic/eval')
            self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Critic/target')

            # 参数置换
            self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                 for ta, ea, tc, ec in
                                 zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

            # actor和critic的eval网络的更新
            with tf.variable_scope(self.name + 'loss'):
                q_target = self.reward + self.GAMMA * self.q_
                # critic的loss
                self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.critic_output)
                # actor的loss
                # self.actor_loss = -tf.reduce_mean(self.critic_output)
                # self.actor_loss = -tf.reduce_mean(self.critic_output+self.L3)
            with tf.variable_scope(self.name + 'train'):
                # 更新actor,3e-5(learning_rate)
                # self.actor_train = tf.train.AdamOptimizer(3e-5).minimize((self.actor_loss), var_list=self.ae_params)
                # 更新critic,3e-5
                self.critic_train = tf.train.AdamOptimizer(3e-4).minimize(self.td_error, var_list=self.ce_params)

    def actor_network(self, name, state, lstm_state_actor, trainable, scope):
        with tf.variable_scope(scope):
            # rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=8, state_is_tuple=True)
            # outputs, (h_c, h_n) = dynamic_rnn(
            #     rnn_cell,
            #     state,
            #     initial_state=lstm_state_actor,
            #     dtype=tf.float32,
            #     time_major=False
            #
            # )
            #
            # if scope == 'eval':
            #     self.state_output_actor_c = h_c  # none,64
            #     self.state_output_actor_h = h_n  # none,64
            # if scope == 'target':
            #     self.state_output_actor_target_c = h_c  # none,64
            #     self.state_output_actor_target_h = h_n  # none,64
            # shape = outputs.get_shape().as_list()  # none,4,64

            L1 = tf.layers.dense(state, 100,
                                 activation=tf.nn.relu,
                                 trainable=trainable,
                                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                 bias_initializer=tf.random_uniform_initializer(minval=3e-5, maxval=3e-2),
                                 name=name + 'l1')
            L2 = tf.layers.dense(L1, 1,
                                 activation=tf.nn.relu,
                                 trainable=trainable,
                                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                 bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                 name=name + 'l2')

            L3 = tf.layers.dense(L2, 1,
                                 activation=tf.nn.relu,  # [0,+无穷]
                                 # 5e-2
                                 # 均匀分布
                                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                 bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                                 trainable=trainable,
                                 name=name + 'l3')
            return L3

    def critic_network(self, cname, state, other_state, action_input, lstm_state_critic, trainable, scope):
        with tf.variable_scope(scope):
            action = action_input
            # state:none,4,2 other_state:none,4,2 action:none,4,1
            state_action = tf.concat([state, other_state, action], axis=2)  # none,4,5
            rnn_cell_critic = tf.nn.rnn_cell.LSTMCell(num_units=8, state_is_tuple=True)
            outputs_critic, (h_c, h_n) = dynamic_rnn(
                rnn_cell_critic,
                state_action,
                initial_state=lstm_state_critic,
                dtype=tf.float32,
                time_major=False
            )

            if scope == 'eval':
                self.state_output_critic_c = h_c  # none,64
                self.state_output_critic_h = h_n  # none,64
            if scope == 'target':
                self.state_output_critic_target_c = h_c  # none,64
                self.state_output_critic_target_h = h_n  # none,64
            shape = outputs_critic.get_shape().as_list()
            critic_input_1 = tf.layers.dense(outputs_critic, 50,
                                             name=cname + 'l1')
            q = tf.layers.dense \
                (critic_input_1, 1,
                 activation=tf.nn.relu,
                 # -1/np.sqrt(360)、3e-3/2
                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 trainable=trainable,
                 name=cname + 'l2')
            # q:none,4,1
            q = tf.nn.relu(q)
            return q

    # 在离散的动作中，可以用greedy策略
    # 在连续的动作中，丰富采样动作最粗暴的方法就是直接往策略输出的动作中加入高斯噪音
    def choose_action(self, state):

        phase_cycle = 100
        state = state.reshape(1, 4, 2)
        # a_p, self.lstm_state_c, self.lstm_state_h = self.sess.run(
        #     [self.L3, self.state_output_actor_c, self.state_output_actor_h], {
        #         self.state: state,
        #         self.c_state_train: self.lstm_state_c,
        #         self.h_state_train: self.lstm_state_h
        #     }
        # )
        a_p = self.sess.run(
            self.L3,  {
                self.state: state,
            }
        )
        if random.random() < self.epsilon:
            a=np.random.choice(4)
            # print('action',action)
            a_p = a_p.reshape( 4,1)
        # phases_ratio = abs(phases_ratio)
        else:
            a_p = a_p.reshape(4, 1)
            a = np.argmax(a_p)

        return a+1,a_p

        # phases_ratio = abs(phases_ratio)
        # phases_ratio = phases_ratio.reshape(4, 1)
        # phases_ratio = softmax(phases_ratio)
        #
        # if np.random.rand() <= self.epsilon:
        #     # 随机生成四个比例和为1
        #     phases_ratio_random = np.random.dirichlet(np.ones(4), size=1)
        #     phases_ratio_random = np.squeeze(phases_ratio_random)
        #     phase1 = int(phases_ratio_random[0] * phase_cycle * 0.8 + (phase_cycle * 0.2) / 4)
        #     phase2 = int(phases_ratio_random[1] * phase_cycle * 0.8 + (phase_cycle * 0.2) / 4)
        #     phase3 = int(phases_ratio_random[2] * phase_cycle * 0.8 + (phase_cycle * 0.2) / 4)
        #     phase4 = 100 - phase1 - phase2 - phase3
        # else:
        #     phase1 = int((phases_ratio[0][0] * phase_cycle * 0.8) + (phase_cycle * 0.2) / 4)
        #     phase2 = int((phases_ratio[1][0] * phase_cycle * 0.8) + (phase_cycle * 0.2) / 4)
        #     phase3 = int((phases_ratio[2][0] * phase_cycle * 0.8) + (phase_cycle * 0.2) / 4)
        #     phase4 = 100 - phase1 - phase2 - phase3

        # return phase1, phase2, phase3, phase4, phases_ratio
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

    def remember(self, state, action, reward, next_state,o_s,o_s_):

        index=self.pointer%MEMORY_CAPACITY
        action=np.array(action)
        action=action.reshape((4,))
        transition=np.hstack((state,next_state,reward,action,o_s[0],o_s_[0]))
        self.memory[index,:]=transition
        self.pointer+=1

    def store_transition(self, s, s_, r, a, s_other, s_other_, type):
        # 如果记录action_probabilty:transition:40---8,8,4,4,8,8,
        transition = np.concatenate((s, s_, r, a, s_other, s_other_), axis=None)
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self):

        save_path = './model/MADRL_model_'+self.name+'/weights_'+self.name
        if os.path.exists(save_path)==False: os.makedirs(save_path)
        self.saver.save(self.sess, save_path)


    def learn(self):
        self.sess.run(self.soft_replace)
        bt = 16
        end = self.pointer
        start = end - bt
        # indices = np.random.choice(range(0, end), size=bt)
        # indices = indices % MEMORY_CAPACITY
        # #非随机的
        batch = self.memory[start:end, :]
        # 所有行的0-cross_state_shape列
        past_s = batch[:, 0:cross_state_shape]
        now_s = batch[:, cross_state_shape:cross_state_shape * 2]
        reward = batch[:, cross_state_shape * 2:cross_state_shape * 2 + 4]
        action = batch[:, cross_state_shape * 2 + 4:(cross_state_shape * 2) + 8]
        other_s = batch[:, -cross_state_shape * 2:-cross_state_shape]
        other_s_ = batch[:, -cross_state_shape:]

        past_s = np.array(past_s).reshape(bt, 4, 2)
        now_s = np.array(now_s).reshape(bt, 4, 2)
        reward = np.array(reward).reshape(bt, 4, 1)
        action = np.array(action).reshape(bt, 4, 1)
        # action = np.array(action).reshape(bt, 1,1)
        other_s = np.array(other_s).reshape(bt, 4, 2)
        other_s_ = np.array(other_s_).reshape(bt, 4, 2)

        # train的是eval网络，输入也是与eval相关的输入
        # 与actor相关的历史也是只在选择动作时使用，在learn时用到的历史都是critic的eval和target历史
        # lstm随机更新的方式，在每一次训练前，将隐藏状态置为0
        lstm_state_critic_eval_c, lstm_state_critic_eval_h = self.initial_zero_state_batch, self.initial_zero_state_batch
        lstm_state_critic_target_c, lstm_state_critic_target_h = self.sess.run(
            [self.state_output_critic_target_c, self.state_output_critic_target_h],
            {
                self.state_: now_s,
                self.other_state_: other_s_,
                self.L3: action,
                self.c_state_critic_target: self.initial_zero_state_batch,
                self.h_state_critic_target: self.initial_zero_state_batch
            }
        )

        # lstm_state_actor_c, lstm_state_actor_h = self.sess.run(
        #     [self.state_output_actor_c, self.state_output_actor_h],
        #     {
        #         self.state: past_s,
        #         self.c_state_train: self.initial_zero_state_batch,
        #         self.h_state_train: self.initial_zero_state_batch
        #     }
        # )
        lstm_state_critic_eval_c, lstm_state_critic_eval_h = self.sess.run(
            [self.state_output_critic_c, self.state_output_critic_h],
            {self.state: past_s,
             self.other_state: other_s,
             self.L3: action,
             self.c_state_critic_eval: lstm_state_critic_eval_c,  # zero_state
             self.h_state_critic_eval: lstm_state_critic_eval_h})  # zero_state

        lstm_state_critic_target_c, lstm_state_critic_target_h = self.sess.run(
            [self.state_output_critic_target_c, self.state_output_critic_target_h],
            {
                self.state_: now_s,
                self.other_state_: other_s_,
                self.L3: action,
                self.c_state_critic_target: lstm_state_critic_target_c,
                self.h_state_critic_target: lstm_state_critic_target_h
            }
        )

        # self.sess.run(self.actor_train, {self.state: past_s, self.other_state: other_s, self.L3: action,
        #                                  self.c_state_critic_eval: lstm_state_critic_eval_c,
        #                                  self.h_state_critic_eval: lstm_state_critic_eval_c,
        #                                  self.c_state_train: lstm_state_actor_c,
        #                                  self.h_state_train: lstm_state_actor_h})  # 输入的是past_s,输入的lstm_state是eval的
        _,critic_output = self.sess.run([self.critic_train,self.critic_output],
                      {self.state: past_s, self.L3: action, self.other_state: other_s, self.other_state_: other_s_,
                       self.c_state_critic_eval: lstm_state_critic_eval_c,
                       self.h_state_critic_eval: lstm_state_critic_eval_c,
                       self.c_state_critic_target: lstm_state_critic_target_c,
                       self.h_state_critic_target: lstm_state_critic_target_h,
                       self.reward: reward,
                       self.state_: now_s})
        # critic_loss = self.sess.run(self.td_error,
        #                             {self.state: past_s,
        #                              self.L3: action,
        #                              self.other_state: other_s,
        #                              self.other_state_: other_s_,
        #                              self.reward: reward,
        #                              self.state_: now_s})
        #
        # return critic_loss



