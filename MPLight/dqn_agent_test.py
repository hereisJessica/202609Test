import random
import numpy as np
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
import tensorflow as tf
import random
import os
all_cross=12
all_state=8

par_min=1
par_max=3

ZNlayer=5


MEMORY_CAPACITY=90000
each_store=24
class DQNAgent:
    def __init__(self,name,config,args):
        self.sess = tf.Session()

        self.name=name
        self.state_size = config['state_size']
        self.action_size = 4

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        # self.lr=0.000079
        self.lr=3e-5
        self.epsilon_decay = 0.995
        # self.learning_rate = 0.001
        self.update_target_freq = 5
        self.batch_size = 30

        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.memory = deque(maxlen=MEMORY_CAPACITY)

        self.phase_num=4
        self.phase_state=2


        self._build_net()
        self.saver = tf.train.Saver()
        # self.update_target_network()

        intersection_id = list(config['lane_phase_info'].keys())[0]
        self.phase_list = config['lane_phase_info'][intersection_id]['phase']

        self.action_store=[]
        self.phase_pointer = 0
        self.state=None
        self.need_to_change=False
        self.phase_number=0
        self.epsilon=1
        self.pointer=0

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_net(self):



        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None,4,2], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None,4,2], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None,4,1 ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None,1,1 ], name='a')  # input Action
        self.q_target=tf.placeholder(tf.float32, [None,4,1], name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            e1 = tf.layers.dense(self.s,48,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
            self.q_eval = tf.layers.dense(e1, 1,kernel_initializer=tf.random_uniform_initializer(minval=3e-8, maxval=3e-5),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2), reuse=tf.AUTO_REUSE,name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):

            t1 = tf.layers.dense(self.s_, 48,reuse=tf.AUTO_REUSE,activation=tf.nn.relu)
            self.q_next = tf.layers.dense(t1, 1,kernel_initializer=tf.random_uniform_initializer(minval=3e-8, maxval=3e-5),bias_initializer=tf.random_uniform_initializer(minval=3e-3, maxval=3e-2),reuse=tf.AUTO_REUSE, name='q_')



        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            # self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())


    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward_p, next_state):

        state = state.reshape(1, 8)
        action = action.reshape(1, 4)
        reward_p = reward_p.reshape(1, 4)
        next_state = next_state.reshape(1, 8)

        if self.pointer <= self.MEMORY_CAPACITY:
            self.memory.append((state, action, reward_p, next_state))

        else:
            self.memory.pop()
            self.memory.append((state, action, reward_p, next_state))
        self.pointer += 1

        # index=self.pointer%MEMORY_CAPACITY
        # action=np.array(action)
        # action=action.reshape((4,))
        # transition=np.hstack((state,next_state,reward,action,o_s[0],o_s_[0]))
        # self.memory[index,:]=transition
        # self.pointer+=1

        # self.memory.append((state, action, reward, next_state))
        #
        # self.pointer +=1


    def choose_action(self, state):


        # state=state.reshape(1,1,8)

        action_probability = self.sess.run(self.q_eval, {self.s: state})
        #
        # if np.random.rand() <= self.epsilon:
        #     # a=random.randrange(self.action_size)
        #     return random.randrange(0, 4)+1, action_probability   # 输出0-2
        # # if random.random() < self.epsilon:
        # #
        # #     p_random=[0.01]*self.phase_num
        # #     action=random.randrange(0,4)
        # #     p_random[action]=0.6+p_random[action]
        # #     p_random=np.array(p_random).reshape(1,-1)
        # #     # print('random{}'.format(action))
        # #     return action,p_random # 输出0-2
        # else:

        action_probability=action_probability.reshape(1,4)

        # bt = self.memory[end-1]
        # reward = bt[2][1]
        # # reward=reward*30
        # action_probability =(0.13*reward) + action_probability

        action= np.argmax(action_probability,axis=1)[0]
        # print('unram',action)

        return action+1,action_probability

    def learn(self):

        end = self.pointer
        start = end - 10
        indices = np.random.choice(range(start, end),size=self.batch_size)



        past_s,  action, reward,now_s = self.getmemory(self.memory, indices)

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: now_s,  # fixed params
                self.s: past_s,  # newest params
            })
        q_target=reward + self.gamma*q_next
        _, loss = self.sess.run([self._train_op, self.loss], feed_dict={
            self.s: past_s,
            self.q_target: q_target
        })
        # print(loss)

    def getmemory(self,memory_deque,index):

        s = []
        s_ = []
        a = []
        r = []

        for i in index:
            s.append(memory_deque[i][0])
            r.append(memory_deque[i][1])
            a.append(memory_deque[i][2])
            s_.append(memory_deque[i][3])

        s = np.reshape(s, (len(index), self.phase_num, self.phase_state))
        s_ = np.reshape(s_, (len(index), self.phase_num, self.phase_state))
        a = np.reshape(a, (len(index), self.phase_num, 1))
        r = np.reshape(r, (len(index), self.phase_num, 1))


        return s,a,r,s_
        # return s,s_,a,r,other_s,other_s_


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
        meta_path = 'weights_' + '.meta'
        this_mata_path = os.path.join(model_path, meta_path)
        self.saver = tf.train.import_meta_graph(this_mata_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def save(self,time):

        current_path = os.path.dirname(os.path.abspath(__file__))
        # up_levels_directory = os.path.abspath(os.path.join(current_path, './'))  # 获取上级路径
        model_path='model/MPLight_model_'+time+'/'+self.name+'/weights_'
        save_path = os.path.join(current_path,model_path)
        if not os.path.exists(save_path):os.makedirs(save_path)
        self.saver.save(self.sess,save_path)

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
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
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
# tf_export('linalg.slogdet')(slogdet)
diag = array_ops.matrix_diag
diag_part = array_ops.matrix_diag_part
eigh = linalg_ops.self_adjoint_eig
eigvalsh = linalg_ops.self_adjoint_eigvals
einsum = special_math_ops.einsum
eye = linalg_ops.eye
inv = linalg_ops.matrix_inverse
logm = gen_linalg_ops.matrix_logarithm
lu = gen_linalg_ops.lu
# tf_export('linalg.logm')(logm)
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



