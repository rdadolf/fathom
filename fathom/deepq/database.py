import numpy as np
import gc
import time
import cv2

class database:
	def __init__(self, params):
		self.size = params['db_size']
		self.img_scale = params['img_scale']
		self.states = np.zeros([self.size,84,84],dtype='uint8') #image dimensions
		self.actions = np.zeros(self.size,dtype='float32')
		self.terminals = np.zeros(self.size,dtype='float32')
		self.rewards = np.zeros(self.size,dtype='float32')
		self.bat_size = params['batch']
		self.bat_s = np.zeros([self.bat_size,84,84,4])
		self.bat_a = np.zeros([self.bat_size])
		self.bat_t = np.zeros([self.bat_size])
		self.bat_n = np.zeros([self.bat_size,84,84,4])
		self.bat_r = np.zeros([self.bat_size])

		self.counter = 0 #keep track of next empty state
		self.flag = False
		return

	def get_batches(self):		
		for i in range(self.bat_size):
			idx = 0
			while idx < 3 or (idx > self.counter-2 and idx < self.counter+3):
				idx = np.random.randint(3,self.get_size()-1)
			self.bat_s[i] = np.transpose(self.states[idx-3:idx+1,:,:],(1,2,0))/self.img_scale
			self.bat_n[i] = np.transpose(self.states[idx-2:idx+2,:,:],(1,2,0))/self.img_scale
			self.bat_a[i] = self.actions[idx]
			self.bat_t[i] = self.terminals[idx]
			self.bat_r[i] = self.rewards[idx]
		#self.bat_s[0] = np.transpose(self.states[10:14,:,:],(1,2,0))/self.img_scale
		#self.bat_n[0] = np.transpose(self.states[11:15,:,:],(1,2,0))/self.img_scale
		#self.bat_a[0] = self.actions[13]
		#self.bat_t[0] = self.terminals[13]
		#self.bat_r[0] = self.rewards[13]

		return self.bat_s,self.bat_a,self.bat_t,self.bat_n,self.bat_r

	def insert(self, prevstate_proc,reward,action,terminal):
		self.states[self.counter] = prevstate_proc
		self.rewards[self.counter] = reward
		self.actions[self.counter] = action
		self.terminals[self.counter] = terminal
		#update counter
		self.counter += 1
		if self.counter >= self.size:
			self.flag = True
			self.counter = 0
		return

	def get_size(self):
		if self.flag == False:
			return self.counter
		else:
			return self.size
	    
