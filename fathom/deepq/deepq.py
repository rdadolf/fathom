#!/usr/bin/env python
# NOTE: Based on Tejas Kulkarni's implementation
# (https://github.com/mrkulk/deepQN_tensorflow).
import tensorflow as tf

from fathom.nn import GenericModel, default_runstep

from database import *
from emulator import *
import tensorflow as tf
import numpy as np
import time
import cv2
import datetime

# TODO: clean up this file
nature_params = {
  'game': 'breakout',
  'window_name': "NNModel: Deep Q-Learning for Atari",
  'frameskip': 1,
  'visualize' : False,
  'network_type':'nips',
  'ckpt_file':None,
  'steps_per_epoch': 50000,
  'num_epochs': 100,
  'eval_freq':50000,
  'steps_per_eval':10000,
  'copy_freq' : 10000,
  'disp_freq':10000,
  'save_interval':10000,
  'db_size': 1000000,
  'batch': 32,
  'num_act': 0,
  'input_dims' : [210, 160, 3],
  'input_dims_proc' : [84, 84, 4],
  'learning_interval': 1,
  'eps': 1.0,
  'eps_step':1000000,
  'eps_min' : 0.1,
  'eps_eval' : 0.05,
  'discount': 0.95,
  'lr': 0.0002,
  'rms_decay':0.99,
  'rms_eps':1e-6,
  'train_start':100, # default: 100
  'img_scale':255.0,
  'clip_delta' : 0, #nature : 1
  'gpu_fraction' : 0.25,
  'batch_accumulator':'mean',
  'record_eval' : True,
  'only_eval' : 'n'
}

nature_params['steps_per_epoch']= 200000
nature_params['eval_freq'] = 100000
nature_params['steps_per_eval'] = 10000
nature_params['copy_freq'] = 10000
nature_params['disp_freq'] = 20000
nature_params['save_interval'] = 20000
#nature_params['learning_interval'] = 1
nature_params['discount'] = 0.99
nature_params['lr'] = 0.00025
nature_params['rms_decay'] = 0.95
nature_params['rms_eps']=0.01
nature_params['clip_delta'] = 1.0
#nature_params['train_start']=50000
nature_params['batch_accumulator'] = 'sum'
nature_params['eps_step'] = 1000000
nature_params['num_epochs'] = 250
nature_params['batch'] = 32

# The actual neural network interface implementation is the network which
# combines the Q-network and target-network below, not this one.
class DeepQNetNature(object):
  """Q-learning network which approximates action-value and action-value targets."""
  def __init__(self, params, parent_graph):
    self.G = parent_graph
    self.build(params)

  def build(self, params):
    with self.G.as_default():
      self.network_type = 'nature'
      self.params = params
      self.network_name = "deepqnet"
      self.x = tf.placeholder('float32',[None,84,84,4],name=self.network_name + '_x')
      self.q_t = tf.placeholder('float32',[None],name=self.network_name + '_q_t')
      self.actions = tf.placeholder("float32", [None, params['num_act']],name=self.network_name + '_actions')
      self.rewards = tf.placeholder("float32", [None],name=self.network_name + '_rewards')
      self.terminals = tf.placeholder("float32", [None],name=self.network_name + '_terminals')

      #conv1
      layer_name = 'conv1' ; size = 8 ; channels = 4 ; filters = 32 ; stride = 4
      self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
      self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
      self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='VALID',name=self.network_name + '_'+layer_name+'_convs')
      self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations')
      #self.n1 = tf.nn.lrn(self.o1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      #conv2
      layer_name = 'conv2' ; size = 4 ; channels = 32 ; filters = 64 ; stride = 2
      self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
      self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
      self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='VALID',name=self.network_name + '_'+layer_name+'_convs')
      self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations')
      #self.n2 = tf.nn.lrn(self.o2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      #conv3
      layer_name = 'conv3' ; size = 3 ; channels = 64 ; filters = 64 ; stride = 1
      self.w3 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
      self.b3 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
      self.c3 = tf.nn.conv2d(self.o2, self.w3, strides=[1, stride, stride, 1], padding='VALID',name=self.network_name + '_'+layer_name+'_convs')
      self.o3 = tf.nn.relu(tf.add(self.c3,self.b3),name=self.network_name + '_'+layer_name+'_activations')
      #self.n2 = tf.nn.lrn(self.o2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      #flat
      o3_shape = self.o3.get_shape().as_list()

      #fc3
      layer_name = 'fc4' ; hiddens = 512 ; dim = o3_shape[1]*o3_shape[2]*o3_shape[3]
      self.o3_flat = tf.reshape(self.o3, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat')
      self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
      self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
      self.ip4 = tf.add(tf.matmul(self.o3_flat,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_ips')
      self.o4 = tf.nn.relu(self.ip4,name=self.network_name + '_'+layer_name+'_activations')

      #fc4
      layer_name = 'fc5' ; hiddens = params['num_act'] ; dim = 512
      self.w5 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
      self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
      self.y = tf.add(tf.matmul(self.o4,self.w5),self.b5,name=self.network_name + '_'+layer_name+'_outputs')

      #Q,Cost,Optimizer
      self.discount = tf.constant(self.params['discount'])
      self.yj = tf.add(self.rewards, tf.mul(1.0-self.terminals, tf.mul(self.discount, self.q_t)))
      self.Qxa = tf.mul(self.y,self.actions)
      self.Q_pred = tf.reduce_max(self.Qxa, reduction_indices=1)
      #self.yjr = tf.reshape(self.yj,(-1,1))
      #self.yjtile = tf.concat(1,[self.yjr,self.yjr,self.yjr,self.yjr])
      #self.yjax = tf.mul(self.yjtile,self.actions)

      #half = tf.constant(0.5)
      self.diff = tf.sub(self.yj, self.Q_pred)
      if self.params['clip_delta'] > 0 :
        self.quadratic_part = tf.minimum(tf.abs(self.diff), tf.constant(self.params['clip_delta']))
        self.linear_part = tf.sub(tf.abs(self.diff),self.quadratic_part)
        self.diff_square = 0.5 * tf.pow(self.quadratic_part,2) + self.params['clip_delta']*self.linear_part

      else:
        self.diff_square = tf.mul(tf.constant(0.5),tf.pow(self.diff, 2))
      # add optimization

      self.loss()
      self.train()

  def loss(self):
    with self.G.as_default():
      if self.params['batch_accumulator'] == 'sum':
        self.cost = tf.reduce_sum(self.diff_square)
      else:
        self.cost = tf.reduce_mean(self.diff_square)

  def train(self):
    with self.G.as_default():
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)
      return self.rmsprop

class DeepQ(GenericModel):
  """Deep Q-Learning."""
  forward_only = False

  def __init__(self, device=None, init_options=None, game=nature_params['game']):
    super(DeepQ,self).__init__(device=device, init_options=init_options)
    assert game in ["breakout", "space_invaders", "seaquest"]

    self.G = tf.Graph()

    # NOTE: moved tf.Graph construction to setup
    self.params = nature_params

    self.DB = database(self.params)
    self.engine = emulator(rom_name='{}.bin'.format(game), vis=self.params['visualize'], frameskip=self.params['frameskip'], windowname=self.params['window_name'])
    #self.engine = emulator(rom_name='{}.bin'.format(game), vis=self.params['visualize'], frameskip=self.params['frameskip'], windowname=self.params['window_name'])
    self.params['num_act'] = len(self.engine.legal_actions)

    with self.G.device(device):
      self.build_inference()

  def build_inference(self):
    with self.G.as_default():
      print 'Building QNet and targetnet...'
      self.qnet = DeepQNetNature(self.params, self.G)
      self.targetnet = DeepQNetNature(self.params, self.G)
      saver_dict = {'qw1':self.qnet.w1,'qb1':self.qnet.b1,
                    'qw2':self.qnet.w2,'qb2':self.qnet.b2,
                    'qw3':self.qnet.w3,'qb3':self.qnet.b3,
                    'qw4':self.qnet.w4,'qb4':self.qnet.b4,
                    'qw5':self.qnet.w5,'qb5':self.qnet.b5,
                    'tw1':self.targetnet.w1,'tb1':self.targetnet.b1,
                    'tw2':self.targetnet.w2,'tb2':self.targetnet.b2,
                    'tw3':self.targetnet.w3,'tb3':self.targetnet.b3,
                    'tw4':self.targetnet.w4,'tb4':self.targetnet.b4,
                    'tw5':self.targetnet.w5,'tb5':self.targetnet.b5,
                    'step':self.qnet.global_step}

      print("#ops", len(self.G.get_operations()))

      self.saver = tf.train.Saver(saver_dict)
      #self.saver = tf.train.Saver()

      self.cp_ops = [
        self.targetnet.w1.assign(self.qnet.w1),self.targetnet.b1.assign(self.qnet.b1),
        self.targetnet.w2.assign(self.qnet.w2),self.targetnet.b2.assign(self.qnet.b2),
        self.targetnet.w3.assign(self.qnet.w3),self.targetnet.b3.assign(self.qnet.b3),
        self.targetnet.w4.assign(self.qnet.w4),self.targetnet.b4.assign(self.qnet.b4),
        self.targetnet.w5.assign(self.qnet.w5),self.targetnet.b5.assign(self.qnet.b5)]

      if self.params['ckpt_file'] is not None:
        print 'loading checkpoint : ' + self.params['ckpt_file']
        self.saver.restore(self.sess,self.params['ckpt_file'])
        temp_train_cnt = self.sess.run(self.qnet.global_step)
        temp_step = temp_train_cnt * self.params['learning_interval']
        print 'Continue from'
        print '        -> Steps : ' + str(temp_step)
        print '        -> Minibatch update : ' + str(temp_train_cnt)

  def model(self):
    return self.G

  def setup(self, setup_options=None):
    super(DeepQ,self).setup(setup_options=setup_options)
    with self.G.as_default():
      if setup_options is None:
        self.setup_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_fraction']))
      else:
        self.setup_config = tf.ConfigProto(**setup_options)
        self.setup_config.gpu_options.per_process_gpu_memory_fraction=self.params['gpu_fraction']

      self.sess = tf.Session(config=self.setup_config)
      self.init = tf.initialize_all_variables()
      self.sess.run(self.init)
      self.sess.run(self.cp_ops)

      self.reset_game()
      self.step = 0
      self.reset_statistics('all')
      self.train_cnt = self.sess.run(self.qnet.global_step)

  def reset_game(self):
    self.state_proc = np.zeros((84,84,4)); self.action = -1; self.terminal = False; self.reward = 0
    self.state = self.engine.newGame()
    self.state_resized = cv2.resize(self.state,(84,110))
    self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
    self.state_gray_old = None
    self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']

  def reset_statistics(self, mode):
    if mode == 'all':
      self.epi_reward_train = 0
      self.epi_Q_train = 0
      self.num_epi_train = 0
      self.total_reward_train = 0
      self.total_Q_train = 0
      self.total_cost_train = 0
      self.steps_train = 0
      self.train_cnt_for_disp = 0
    self.step_eval = 0
    self.epi_reward_eval = 0
    self.epi_Q_eval = 0
    self.num_epi_eval = 0
    self.total_reward_eval = 0
    self.total_Q_eval = 0

  def select_action(self, st, runstep=None):
    with self.G.as_default():
      if np.random.rand() > self.params['eps']:
        #greedy with random tie-breaking
        if not self.forward_only:
          Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
        else:
          Q_pred = runstep(self.sess, self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]

        a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
        if len(a_winner) > 1:
          act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
          return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
        else:
          act_idx = a_winner[0][0]
          return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
      else:
        #random
        act_idx = np.random.randint(0,len(self.engine.legal_actions))
        if not self.forward_only:
          Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
        else:
          Q_pred = runstep(self.sess, self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
        return act_idx,self.engine.legal_actions[act_idx], Q_pred[act_idx]

  def get_onehot(self,actions):
    actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))

    for i in range(self.params['batch']):
      actions_onehot[i,int(actions[i])] = 1
    return actions_onehot

  def run(self, runstep=default_runstep, n_steps=1):
    self.s = time.time()
    print self.params
    print 'Start training!'
    print 'Collecting replay memory for ' + str(self.params['train_start']) + ' steps'

    with self.G.as_default():
      while self.step < (self.params['steps_per_epoch'] * self.params['num_epochs'] * self.params['learning_interval'] + self.params['train_start']):
        if not self.forward_only:
          if self.step >= n_steps:
            return
          if self.DB.get_size() >= self.params['train_start'] : self.step += 1 ; self.steps_train += 1
        else:
          if self.step_eval >= n_steps:
            return
        self.step_eval += 1
        if self.state_gray_old is not None and not self.forward_only:
          self.DB.insert(self.state_gray_old[26:110,:],self.reward_scaled,self.action_idx,self.terminal)

        if not self.forward_only and self.params['copy_freq'] > 0 and self.step % self.params['copy_freq'] == 0 and self.DB.get_size() > self.params['train_start']:
          print '&&& Copying Qnet to targetnet\n'
          self.sess.run(self.cp_ops)

        if not self.forward_only and self.step % self.params['learning_interval'] == 0 and self.DB.get_size() > self.params['train_start'] :
          bat_s,bat_a,bat_t,bat_n,bat_r = self.DB.get_batches()
          bat_a = self.get_onehot(bat_a)

          if self.params['copy_freq'] > 0 :
            feed_dict={self.targetnet.x: bat_n}
            q_t = self.sess.run(self.targetnet.y,feed_dict=feed_dict)
          else:
            feed_dict={self.qnet.x: bat_n}
            q_t = self.sess.run(self.qnet.y,feed_dict=feed_dict)

          q_t = np.amax(q_t,axis=1)

          feed_dict={self.qnet.x: bat_s, self.qnet.q_t: q_t, self.qnet.actions: bat_a, self.qnet.terminals:bat_t, self.qnet.rewards: bat_r}

          # NOTE: we only runstep the Qnet
          _,self.train_cnt,self.cost = runstep(self.sess, [self.qnet.rmsprop,self.qnet.global_step,self.qnet.cost],feed_dict=feed_dict)

          self.total_cost_train += np.sqrt(self.cost)
          self.train_cnt_for_disp += 1

        if not self.forward_only:
          self.params['eps'] = max(self.params['eps_min'],1.0 - float(self.train_cnt * self.params['learning_interval'])/float(self.params['eps_step']))
        else:
          self.params['eps'] = 0.05

        if self.DB.get_size() > self.params['train_start'] and self.step % self.params['save_interval'] == 0 and not self.forward_only:
          save_idx = self.train_cnt
          self.saver.save(self.sess,'ckpt/model_'+self.params['network_type']+'_'+str(save_idx))
          sys.stdout.write('$$$ Model saved : %s\n\n' % ('ckpt/model_'+self.params['network_type']+'_'+str(save_idx)))
          sys.stdout.flush()

        if not self.forward_only and self.step > 0 and self.step % self.params['eval_freq'] == 0 and self.DB.get_size() > self.params['train_start']:
          self.reset_game()
          if self.step % self.params['steps_per_epoch'] == 0 : self.reset_statistics('all')
          else: self.reset_statistics('eval')
          self.forward_only = True
          #TODO : add video recording
          continue
        if not self.forward_only and self.step > 0 and self.step % self.params['steps_per_epoch'] == 0 and self.DB.get_size() > self.params['train_start']:
          self.reset_game()
          self.reset_statistics('all')
          #self.forward_only = True
          continue

        if self.forward_only and self.step_eval >= self.params['steps_per_eval'] :
          self.reset_game()
          self.reset_statistics('eval')
          self.forward_only = False
          continue

        if self.terminal:
          self.reset_game()
          if not self.forward_only:
            self.num_epi_train += 1
            self.total_reward_train += self.epi_reward_train
            self.epi_reward_train = 0
          else:
            self.num_epi_eval += 1
            self.total_reward_eval += self.epi_reward_eval
            self.epi_reward_eval = 0
          continue

        self.action_idx,self.action, self.maxQ = self.select_action(self.state_proc, runstep=runstep)
        self.state, self.reward, self.terminal = self.engine.next(self.action)
        self.reward_scaled = self.reward // max(1,abs(self.reward))
        if not self.forward_only : self.epi_reward_train += self.reward ; self.total_Q_train += self.maxQ
        else : self.epi_reward_eval += self.reward ; self.total_Q_eval += self.maxQ

        self.state_gray_old = np.copy(self.state_gray)
        self.state_proc[:,:,0:3] = self.state_proc[:,:,1:4]
        self.state_resized = cv2.resize(self.state,(84,110))
        self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
        self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']

        print("Finished step {0} ({1})".format(self.step_eval, datetime.datetime.now()))

  @property
  def loss(self):
    return self.qnet.cost

  @property
  def train(self):
    return self.qnet.rmsprop

  @property
  def labels(self):
    return

  @property
  def inputs(self):
    return self.qnet.x, self.qnet.q_t, self.qnet.actions, self.qnet.rewards, self.qnet.terminals

  @property
  def outputs(self):
    return self.qnet.y # just outputs, not predictions

  def teardown(self):
    if self.sess is not None:
      self.sess.close()
      self.sess = None

class DeepQFwd(DeepQ):
  forward_only = True

if __name__=='__main__':
  m = DeepQ()
  m.setup()
  m.run(runstep=default_runstep, n_steps=100)
  m.teardown()

