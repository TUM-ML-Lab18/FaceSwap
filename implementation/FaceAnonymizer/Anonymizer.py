#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#

import torch
from torch.utils.data import DataLoader
from models import Encoder, Decoder1, Decoder2

class Anonymizer():

	def __init__(self, data1, data2, batch_size=50, epochs=500, learning_rate = 1e-4):
		self.encoder = Encoder.model
		self.decoder1 = Decoder1.model
		self.decoder2 = Decoder2.model
		self.lossfn = torch.nn.MSELoss(size_average=False)
		self.dataLoader1 = DataLoader(data1, batch_size, shuffle=True, num_workers=4)
		self.dataLoader2 = DataLoader(data2, batch_size, shuffle=True, num_workers=4)
		self.epochs = epochs
	
	def train(self):
		for i_epoch in range(epochs):
			# first face
			for i_batch, sample in enumerate(self.dataLoader1):
				pass
			
			# second face
			for i_batch, sample in enumerate(self.dataLoader2):
				pass

	def optimize(self):
		pass

	def save_model(self):
		pass

	def load_model(self):
		pass