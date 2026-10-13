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
import os
all_state=8
all_state_e=8
all_cross=1
global WNM
#WNM=np.random.random(size=(35,1))
WNM=np.random.uniform(low=0,high=0.5,size=(35,4))
MEMORY_CAPACITY=50000
each_store=24#s=8 a=4 r=4 s'=8
state_shape=8
class DQNAgent:
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
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.other_state=None
        self.past_action=None
        #self.sess.run(tf.global_variables_initializer())
    def _build_model(self):

        # par_min=1
        # par_max=3
        par_min=-4
        par_max=-1
        self.s = tf.placeholder(tf.float32, [None,4*2], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None,4*2], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, 4], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, 4], name='a')  # input Action
        self.wnm_e = tf.placeholder(tf.float32, [35,4], name='wnm1')
        self.wnm_t = tf.placeholder(tf.float32, [35, 4], name='wnm2')

        self.q_target = tf.placeholder(tf.float32, [None, 4])
        # self.s_other=tf.placeholder(tf.float32,shape=[None,88],name='s_other')
        # self.s_other_action=tf.placeholder(tf.float32,shape=[None,11],name='s_other_action')
        # self.s_other_store=[]

        with tf.variable_scope('eval_net'):

            e1 = tf.layers.dense(self.s,100,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
            self.q_eval = tf.layers.dense(e1, 4,kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2), reuse=tf.AUTO_REUSE,name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):

            t1 = tf.layers.dense(self.s_, 100,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
            self.q_next = tf.layers.dense(t1, 4,kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),reuse=tf.AUTO_REUSE, name='q_')

        # with tf.variable_scope('eval_net'):
        #
        #     e1 = tf.layers.dense(self.s,100,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
        #     self.q_eval = tf.layers.dense(e1, 4,kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2), reuse=tf.AUTO_REUSE,name='q')
        #
        # # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #
        #     t1 = tf.layers.dense(self.s_, 100,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
        #     self.q_next = tf.layers.dense(t1, 4,kernel_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),reuse=tf.AUTO_REUSE, name='q_')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def remember(self, state, action, reward, next_state):

        index=self.pointer%MEMORY_CAPACITY
        action=np.array(action)
        action=action.reshape((4,))
        transition=np.hstack((state,action,reward,next_state))
        self.memory[index,:]=transition
        self.pointer+=1
    def choose_action(self,state):

        state=state.reshape((1,8))
        action_probability = self.sess.run(self.q_eval, {self.s: state})
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

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, model_path):
        meta_path = 'weights_'+'.meta'
        this_mata_path = os.path.join(model_path,meta_path)
        self.saver = tf.train.import_meta_graph(this_mata_path)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(model_path))


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

        end = self.pointer
        # bt = random.randint(0, end - 2)

        past_s = self.memory[end-8:end-3,0:8]
        past_s = np.reshape(past_s, (5, 8))
        now_s = self.memory[end-8:end-3,16:24]
        now_s = np.reshape(now_s, (5, 8))
        reward = self.memory[end-8:end-3,12:16]
        reward = np.reshape(reward, (5, 4))
        # action = self.memory[end-8:end-3,8:12]
        # action = np.reshape(action, (5, 4))

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: now_s,  # fixed params
                self.s: past_s,  # newest params
            })

        q_target = reward + self.gamma * q_next

        _, loss = self.sess.run([self._train_op, self.loss], feed_dict={
            self.s: past_s,
            self.q_target: q_target
        })
        print(loss)






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
