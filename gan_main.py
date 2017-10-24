#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
import numpy as np
from data import DataFolder
from model import Discriminator, SalGAN
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
data_dirs = [
			 ("/media/mowayao/data/salient/data/ECSSD/train/images", "/media/mowayao/data/salient/data/ECSSD/train/gt"),
             ("/media/mowayao/data/salient/data/MSRA10K/images", "/media/mowayao/data/salient/data/MSRA10K/gt"),
             ("/media/mowayao/data/salient/data/HKU-IS/imgs", "/media/mowayao/data/salient/data/HKU-IS/gt")
             ]

test_dirs = [("/media/mowayao/data/salient/data/ECSSD/test/images", "/media/mowayao/data/salient/data/ECSSD/test/gt")
]
def process_data_dir(data_dir):
	files = os.listdir(data_dir)
	files = map(lambda x: os.path.join(data_dir, x), files)
	return sorted(files)

DATA_DICT = {}

IMG_FILES = []
GT_FILES = []

IMG_FILES_TEST = []
GT_FILES_TEST = []

for dir_pair in data_dirs:
	X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
	IMG_FILES.extend(X)
	GT_FILES.extend(y)

for dir_pair in test_dirs:
	X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
	IMG_FILES_TEST.extend(X)
	GT_FILES_TEST.extend(y)

IMGS_train, GT_train = IMG_FILES, GT_FILES

train_folder = DataFolder(IMGS_train, GT_train, True)

train_data = DataLoader(train_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True)

test_folder = DataFolder(IMG_FILES_TEST, GT_FILES_TEST, False)
test_data = DataLoader(test_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)


Sal = SalGAN().cuda()
Dis = Discriminator(config.IMG_SIZE).cuda()
#Sal._modules = torch.load("checkpoint/model.pth")['net']
optimizer = optim.SGD(
			[{'params': Sal.encoder.parameters(), 'lr': config.BASE_LEARNING_RATE, 'momentum':0.88, 'weight_decay':1e-4},
			 {'params': Sal.decoder.parameters(), 'lr': config.LEARNING_RATE, 'momentum':0.88, 'weight_decay':1e-4},
			 {'params': Dis.parameters(), 'lr': config.LEARNING_RATE, 'momentum':0.88, 'weight_decay':1e-4},
            ])
evaluation = nn.L1Loss()
scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.2)
best_eval = None
for epoch in xrange(1, config.NUM_EPOCHS+1):
	Sal.train()
	sum_train_mae = 0
	sum_train_loss = 0
	sum_train_gan = 0
	##train
	for iter_cnt, (img_batch, label_batch, weights) in enumerate(train_data):

		optimizer.zero_grad()
		img_batch = Variable(img_batch, requires_grad=False).cuda()
		label_batch = Variable(label_batch, requires_grad=False).cuda()
		weights =Variable(weights, requires_grad=False).cuda()
		pred_label = Sal(img_batch)
		fake_out = Dis(torch.cat((img_batch, pred_label.unsqueeze(dim=1)), dim=1))
		real_out = Dis(torch.cat((img_batch, label_batch.unsqueeze(dim=1)), dim=1))
		dis_out = torch.cat((fake_out, real_out), dim=1)
		dis_label = Variable(torch.cat((torch.zeros(fake_out.size()), torch.ones(real_out.size())), dim=1)).cuda()
		gan_loss = F.binary_cross_entropy(dis_out, dis_label) + F.binary_cross_entropy(real_out, Variable(torch.ones(real_out.size())).cuda())
		loss = F.binary_cross_entropy(pred_label, label_batch, weights) + 0.05*gan_loss
		mae = evaluation(pred_label, label_batch)
		sum_train_loss += loss.data[0]
		sum_train_mae += mae.data[0]
		sum_train_gan += gan_loss.data[0]
		loss.backward()
		optimizer.step()

		print "Epoch:{}\t  {}/{}\t loss:{} \t mae:{}".format(epoch, iter_cnt+1,
		                                         len(train_folder)/config.BATCH_SIZE,
		                                         sum_train_loss/(iter_cnt+1),
		                                         sum_train_mae/(iter_cnt+1))

	##evaluate
	Sal.eval()
	sum_eval_mae = 0
	sum_eval_loss = 0
	num_eval = 0
	scheduler.step()
	for iter_cnt, (img_batch, label_batch, weights) in enumerate(test_data):
		img_batch = Variable(img_batch).cuda()
		label_batch = Variable(label_batch).cuda()
		weights = Variable(weights).cuda()
		pred_label = Sal(img_batch)
		loss = F.binary_cross_entropy(pred_label, label_batch, weights)
		#mae = evaluation(pred_label, label_batch)
		#sum_eval_loss += loss.data[0] * img_batch.size(0)
		sum_eval_loss += loss.data[0] * img_batch.size(0)
		num_eval += img_batch.size(0)
	#eval_loss = sum_eval_loss / num_eval
	eval_loss = sum_eval_loss / num_eval
	print "Validation \t loss:{} \t loss:{}".format(epoch,
	                                         #eval_loss,
	                                         eval_loss)
	if best_eval is None or best_eval < eval_loss:
		best_eval = eval_loss
		state = {
			'net': Sal._modules,
			'gan': Dis._modules,
			'mae': best_eval,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/gan_model.pth')
