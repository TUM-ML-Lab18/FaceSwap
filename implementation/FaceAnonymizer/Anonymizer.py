#
# This file includes the main functionality of the FaceAnonymizer module
# Author: Alexander Becker
#

import torch
from models import Encoder, Decoder1, Decoder2

class Anonymizer():

	def __init__(self, data1, data2, batch_size=50, epochs=500, learning_rate = 1e-4):
		self.encoder = Encoder.model
		self.decoder1 = Decoder1.model
		self.decoder2 = Decoder2.model
		self.lossfn = torch.nn.MSELoss(size_average=False)
		self.data1 = data1
		self.data2 = data2
		self.epochs = epochs
	
	def train(self):
		for epoch in range(epochs):
			#TODO: divide into batches
			#TODO: build overall model (how?)

	def optimize(self):
		pass

	def save_model(self):
		pass

	def load_model(self):
		pass