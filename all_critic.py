import tensorflow as tf
import numpy as np
from MADDPG import MADDPG

from collections import deque
BATCH_SIZE=25
MEMORY_CAPACITY=2000
class all_critic():
    def __init__(self,a):

        self.s = tf.placeholder(shape=[None, 9*24], dtype=tf.float32)
        self.s_= tf.placeholder(shape=[None, 9*24], dtype=tf.float32)
        self.a = tf.placeholder(shape=[None, 9 * 1], dtype=tf.float32)
        self.r = tf.placeholder(shape=[None, 9 * 1], dtype=tf.float32)

        self.lr=0.001
        self.gamma=0.9
        TAU=0.01 # soft replacement
        self.s_dim=9*24
        self.a_dim=9
        self.r_dim=9
        self.pointer=0

        self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + self.r_dim), dtype=np.float32)




        self.q = self.critic_network( self.s, self.a, scope='eval')
        self.q_ = self.critic_network(self.s_,self.a, scope='target')

        self.t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        self.e_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')

        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea)]
                             for ta, ea,  in zip(self.t_params, self.e_params)]


        with tf.variable_scope('target_q'):
            self.target_q = self.r + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('all_C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)

        self.sess=tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)





    def critic_network(self,s,a,scope):

        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 414
                w1_s = tf.get_variable('w1_s', [9*24, n_l1], initializer=init_w)
                w1_a = tf.get_variable('w1_a', [9, n_l1], initializer=init_w)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b)
                critic_l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('l2'):
                critic_l2 = tf.layers.dense(critic_l1, 626, name='all_critic_l2')
                critic_l3 = tf.layers.dense(critic_l2, 555, name='all_critic_l3')

            with tf.variable_scope('q'):
                q = tf.layers.dense(critic_l3 , 1, kernel_initializer=init_w, bias_initializer=init_b)   # Q(s,a)
        return q

    def Q(self,s,a):
        return self.sess.run(self.q,feed_dict={self.s:s,self.a:a})

    def Q_(self, s_, a):
        return self.sess.run(self.q_, feed_dict={self.s_: s_, self.a: a})



    def store_transition(self, s, a, r, s_):

        s=np.reshape(s,[1,self.s_dim])
        a=np.reshape(a,[1,self.a_dim])
        r=np.reshape(r,[1,self.r_dim])
        s_=np.reshape(s_,[1,self.s_dim])
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        self.sess.run(self.soft_replace)
        indices = np.random.choice(self.pointer, size=BATCH_SIZE)

        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim:self.s_dim + self.a_dim+self.r_dim]
        bs_ = bt[:, -self.s_dim:]


        self.sess.run(self.train_op, feed_dict={self.s:bs,self.s_:bs_, self.a: ba, self.r: br})






