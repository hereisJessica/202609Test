from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
import tensorflow as tf
all_state=8
all_state_e=8
all_cross=1
global WNM
#WNM=np.random.random(size=(35,1))
WNM=np.random.uniform(low=0,high=0.5,size=(35,4))
MEMORY_CAPACITY=50000
each_store=40#s=8 a=4 r=4 s'=8
state_shape=8
TAU=0.01
class MARDDPG():
    def __init__(self,name,config):
        self.name=name
        self.state_size = config['state_size']#9*12*2
        self.action_size = 4
        #self.memory = deque(maxlen=2000)
        self.memory=np.zeros((MEMORY_CAPACITY,each_store))
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_freq = 5
        self.batch_size = 30
        self.lr=0.001
        self.model = self._build_model()
        #self.target_model = self._build_model()
        #self.update_target_network()

        intersection_id = list(config['lane_phase_info'].keys())[0]
        self.phase_list = config['lane_phase_info'][intersection_id]['phase']

        self.action_store=[]
        self.phase_pointer = 0
        self.pointer=0
        self.state=None
        self.need_to_change=False
        self.phase_number=0
        self.epsilon=1
        self.sess=tf.Session()
        ''' MARDDPG'''

        self.other_state=None
        self.other_state_ = None
        self.past_action=None


        self.sess.run(tf.compat.v1.global_variables_initializer())

        #self.sess.run(tf.global_variables_initializer())


    def _build_model(self):

        # par_min=1
        # par_max=3
        par_min=-4
        par_max=-1
        self.s = tf.placeholder(tf.float32, [None,4*2], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None,4*2], name='s_')  # input Next State
        self.s_o = tf.placeholder(tf.float32, [None,4*2], name='s')  # input State
        self.s_o_ = tf.placeholder(tf.float32, [None,4*2], name='s_')  # input Next State

        self.r = tf.placeholder(tf.float32, [None, 4], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, 4], name='a')  # input Action
        self.wnm_e = tf.placeholder(tf.float32, [35,4], name='wnm1')
        self.wnm_t = tf.placeholder(tf.float32, [35, 4], name='wnm2')

        self.q_target = tf.placeholder(tf.float32, [None, 4])
        # self.s_other=tf.placeholder(tf.float32,shape=[None,88],name='s_other')
        # self.s_other_action=tf.placeholder(tf.float32,shape=[None,11],name='s_other_action')
        # self.s_other_store=[]

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=8)
        self.ha0 = lstm_cell.zero_state([1], np.float32)  # zero是yie tupe type data
        self.hc0 = lstm_cell.zero_state([1], np.float32)


        with tf.variable_scope(self.name + 'Actor'):
            self.L3 = self.actor_network(self.s, scope='eval')
            self.a_ = self.actor_network(self.s_, scope='target')

        with tf.variable_scope(self.name + 'Critic'):
            # 这时输入的action是4*1，改为了1*1
            self.critic_output = self.critic_network(self.name + 'critic', self.s, self.s_o,
                                                     self.L3, scope='eval',
                                                     trainable=tf.AUTO_REUSE)
            self.q_ = self.critic_network(self.name + 'critic_target', self.s_, self.s_o_, self.a_,
                                          scope='target',
                                          trainable=False)

        # 初始化参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + 'Critic/target')

        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in
                             zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]



        with tf.variable_scope('loss'):
            q_target = self.r + self.gamma * self.q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.critic_output)
            # actor的loss
            self.actor_loss = tf.reduce_mean(self.critic_output)
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self.actor_train = tf.train.AdamOptimizer(3e-5).minimize((self.actor_loss), var_list=self.ae_params)
            # 更新critic,3e-5

            self.critic_train = tf.train.AdamOptimizer(3e-4).minimize(-self.td_error, var_list=self.ce_params)

    def actor_network(self,state, scope):

        with tf.variable_scope(scope):

            e1 = tf.layers.dense(state,100,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
            self.q_eval = tf.layers.dense(e1, 4,kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2), reuse=tf.AUTO_REUSE,name='q')
            return self.q_eval

    def critic_network(self, cname,state,s_other,action,trainable,scope):
        with tf.variable_scope(scope):
            state_action=tf.concat([state,action,s_other],axis=1)
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=8)
            state_action, h1 = lstm_cell.__call__(inputs=state_action, state=self.hc0)
            critic_input_1=tf.layers.dense(state_action,360,name=cname+'l1')
            # critic_input_2 = tf.layers.dense(critic_input_1, 72)
            #l1=tf.nn.sigmoid(critic_input_1)
            q = tf.layers.dense\
                (critic_input_1, 4,
                 activation=tf.nn.leaky_relu,
                 kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),
                 trainable=trainable,
                 name=cname+'l2')
            q= tf.nn.relu(q)
            return q


    def remember(self, state, action, reward, next_state,o_s,o_s_):

        index=self.pointer%MEMORY_CAPACITY
        action=np.array(action)
        action=action.reshape((4,))
        transition=np.hstack((state,next_state,reward,action,o_s[0],o_s_[0]))
        self.memory[index,:]=transition
        self.pointer+=1
    def choose_action(self,time, state,all_state,i):

        state=state.reshape((1,8))
        action_probability = self.sess.run(self.L3, {self.s: state})
        action_probability=action_probability[0]
        if random.random() < self.epsilon:
            action=np.random.choice(4)
            # print('action',action)
            action_probability = action_probability.reshape(1, 4)
        else:
 #        action_probability=action_probability.reshape(1,4)

            action= np.argmin(action_probability)
            # print('unram',action)

        return action+1,action_probability

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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

    def learn(self):
        cross_state_shape=8

        self.sess.run(self.soft_replace)
        bt = 1
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

        past_s = np.array(past_s).reshape(bt, 8)
        now_s = np.array(now_s).reshape(bt, 8)
        reward = np.array(reward).reshape(bt, 4)
        action = np.array(action).reshape(bt, 4)
        # action = np.array(action).reshape(bt, 1,1)
        other_s = np.array(other_s).reshape(bt, 8)
        other_s_ = np.array(other_s_).reshape(bt, 8)

        self.sess.run(self.actor_train, {self.s: past_s, self.s_o: other_s, self.L3: action,
                                         # self.c_state_critic_eval:lstm_state_critic_eval_c,
                                         # self.h_state_critic_eval:lstm_state_critic_eval_c,
                                         # self.c_state_train:lstm_state_actor_c,
                                         # self.h_state_train:lstm_state_actor_h
                                         })  # 输入的是past_s,输入的lstm_state是eval的
        self.sess.run(self.critic_train, {self.s: past_s, self.L3: action,
                                          self.s_o: other_s, self.s_o_: other_s_,
                                          self.r: reward,
                                          self.s_: now_s})






from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
band_part = array_ops.matrix_band_part
cholesky = linalg_ops.cholesky
cholesky_solve = linalg_ops.cholesky_solve
det = linalg_ops.matrix_determinant
slogdet = gen_linalg_ops.log_matrix_determinant
#tf_export('linalg.slogdet')(slogdet)
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
logm = gen_linalg_ops.matrix_logarithm
lu = gen_linalg_ops.lu
#tf_export('linalg.logm')(logm)
lstsq = linalg_ops.matrix_solve_ls
norm = linalg_ops.norm
qr = linalg_ops.qr
set_diag = array_ops.matrix_set_diag
solve = linalg_ops.matrix_solve
sqrtm = linalg_ops.matrix_square_root
svd = linalg_ops.svd
tensordot = math_ops.tensordot
trace = math_ops.trace
transpose = array_ops.matrix_transpose
triangular_solve = linalg_ops.matrix_triangular_solve
def _maybe_validate_matrix(a, validate_args):
  """Checks that input is a `float` matrix."""
  assertions = []
  if not a.dtype.is_floating:
    raise TypeError('Input `a` must have `float`-like `dtype` '
                    '(saw {}).'.format(a.dtype.name))
  if a.shape is not None and a.shape.rank is not None:
    if a.shape.rank < 2:
      raise ValueError('Input `a` must have at least 2 dimensions '
                       '(saw: {}).'.format(a.shape.rank))
  elif validate_args:
    assertions.append(
        check_ops.assert_rank_at_least(
            a, rank=2, message='Input `a` must have at least 2 dimensions.'))
  return assertions
@tf_export('linalg.pinv')
def pinv(a, rcond=None, validate_args=False, name=None):
  """Compute the Moore-Penrose pseudo-inverse of one or more matrices.
  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.
  The pseudo-inverse of a matrix `A`, is defined as: 'the matrix that 'solves'
  [the least-squares problem] `A @ x = b`,' i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]
  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.
  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'pinv'.
  Returns:
    a_pinv: (Batch of) pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.
  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.
  #### Examples
  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tf.linalg..pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tf.linalg..pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```
  #### References
  [1]: G. Strang. 'Linear Algebra and Its Applications, 2nd Ed.' Academic Press,
       Inc., 1980, pp. 139-142.
  """
  with ops.name_scope(name or 'pinv'):
    a = ops.convert_to_tensor(a, name='a')

    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        a = array_ops.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:

      def get_dim_size(dim):
        dim_val = tensor_shape.dimension_value(a.shape[dim])
        if dim_val is not None:
          return dim_val
        return array_ops.shape(a)[dim]

      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = math_ops.cast(
            math_ops.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = ops.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * math_ops.reduce_max(singular_values, axis=-1)
    singular_values = array_ops.where_v2(
        singular_values > array_ops.expand_dims_v2(cutoff, -1), singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = math_ops.matmul(
        right_singular_vectors / array_ops.expand_dims_v2(singular_values, -2),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv



# import random
# import numpy as np
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
#
# class DQNAgent:
#     def __init__(self,name,config):
#         self.name=name
#         self.state_size = config['state_size']
#         self.action_size = 4
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.update_target_freq = 5
#         self.batch_size = 30
#         self.model = self._build_model()
#         self.target_model = self._build_model()
#         self.update_target_network()
#
#         intersection_id = list(config['lane_phase_info'].keys())[0]
#         self.phase_list = config['lane_phase_info'][intersection_id]['phase']
#
#         self.action_store=[]
#         self.phase_pointer = 0
#         self.state=None
#         self.need_to_change=False
#         self.phase_number=0
#         self.epsilon=1
#
#
#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(40, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(40, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse',
#                       optimizer=Adam(lr=self.learning_rate))
#         return model
#
#     def update_target_network(self):
#         weights = self.model.get_weights()
#         self.target_model.set_weights(weights)
#
#     def remember(self, state, action, reward, next_state):
#         # action = self.phase_list.index(action)
#         self.memory.append((state, action, reward, next_state))
#
#     def choose_action(self, state):
#         if np.random.rand() <= self.epsilon:
#             a=random.randrange(self.action_size)
#             return random.randrange(self.action_size)
#         '''
#         get action
#         '''
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])  # returns action
#
#     def replay(self):
#         minibatch = random.sample(self.memory, self.batch_size)
#         for state, action, reward, next_state in minibatch:
#             target = (reward + self.gamma *
#                       np.amax(self.target_model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#     def load(self, name):
#         self.model.load_weights(name)
#
#     def save(self, name):
#         self.model.save_weights(name)
#
#     def phase(self, action, current_time):
#         time = 0
#         need_change_interstion = []
#         if current_time == time:
#             past_p = 0
#         else:
#             past_p = self.action_store[self.phase_pointer - 1]
#
#         if action == past_p:
#             self.need_to_change=False
#         else:
#             self.need_to_change=True
#         self.phase_number=action
#         self.phase_pointer += 1
#
#
#         # return
