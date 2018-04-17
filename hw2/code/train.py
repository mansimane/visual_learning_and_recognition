from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file

import visdom
from logger import *
from test import test_net
try:
    from termcolor import cprint
except ImportError:
    cprint = None

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = logger_v
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
test_imdb_name = 'voc_2007_test'

cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
vis_interval = 5000

start_step = 0
end_step = 50000
lr_decay_steps = {150000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = False
use_tensorboard = False
use_visdom = False
log_grads = False

remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load imdb and create data later
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

test_imdb = get_imdb(test_imdb_name)
# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
else:
    pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)

own_state = net.state_dict()

#Loading layers with different names: Courtesy Ziqiang Feng
for name, param in pret_net.items():
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state_name = name
        if name.startswith('classifier.1'):
            own_state_name = name.replace('classifier.1', 'classifier.0', 1)
        elif name.startswith('classifier.4'):
            own_state_name = name.replace('classifier.4', 'classifier.3', 1)
        own_state[own_state_name].copy_(param)
        print('Copied {} to {}'.format(name, own_state_name))
    except:
        print('Did not find {}'.format(name))
        continue
    
#     if name == 'classifier.1.weight':
#         param = param.data
#         own_state['classifier.0.weight'].copy_(param)
#         print('Copied {}'.format(name))
#     if name == 'classifier.1.bias':
#         param = param.data
#         own_state['classifier.0.bias'].copy_(param)
#         print('classifier.0.bias Copied {}'.format(name))
    
#     if name == 'classifier.4.weight':
#         param = param.data
#         own_state['classifier.3.weight'].copy_(param)
#         print('classifier.3.weight Copied {}'.format(name))
    
#     if name == 'classifier.4.bias':
#         param = param.data
#         own_state['classifier.3.bias'].copy_(param)
#         print('Copied {}'.format(name))
    
#     if name == 'classifier.6.weight':#*** necessary?
#         param = param.data
#         own_state['score_cls.0.weight'].copy_(param)
#         own_state['score_det.0.weight'].copy_(param)
#         print('Copied score_cls.0.weight  {}'.format(name))
       
         
        
                
#from IPython.core.debugger import Tracer; Tracer()()

# Move model to GPU and set train mode
net.cuda()
net.train()


# Create optimizer for network parameters
params = list(net.parameters())
optimizer = torch.optim.SGD(params[2:], lr=lr, 
                            momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

logger_v = visdom.Visdom(server='http://localhost' ,port='8099')
logger_t = Logger('./tboard', name='wsddn')
plotter = VisdomLinePlotter(env_name='main_wsddn_train')

for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    #from IPython.core.debugger import Tracer; Tracer()() #labels may be none
    im_data = blobs['data']#1xhxwx3
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']

    # forward
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps, lr, momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every N iterations (N defined in handout)
    if step%5 ==0:   #Plot loss#500
        logger_t.scalar_summary(tag= 'loss', value= loss.data[0], step= step)
        #logger_v.scalar_summary(tag= 'loss', value= loss.data[0], step= step)
        plotter.plot('train_loss', 'train', step, loss.data[0])

    if step%2000 ==0:   #Plot mAP on histograms of weights and gradients
        logger_t.model_param_histo_summary(net, step=step)
    if (step)%5000 ==0 and (step != 0):   #Plot mAP on test/ and classwise APs#5000
        net.eval()
        aps = test_net(name='wsddn_test', net=net, imdb =test_imdb, max_per_image=300, thresh=0.0001, visualize=True, logger=logger_t, step=step)
        mean_ap = np.mean(aps)
        #from IPython.core.debugger import Tracer; Tracer()()

        plotter.plot('wsddn_mAP', 'test', step, mean_ap)
        for idx in range(len(aps)):
            tag = 'ap_' + imdb.classes[idx] + '_ap'
            logger_t.scalar_summary(tag= tag, value= aps[idx], step= step)
        logger_t.scalar_summary(tag= 'wsddn_mAP', value= mean_ap, step= step)  
        net.train()
    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
    if visualize and step%vis_interval==0:
        #TODO: Create required visualizations
        if use_tensorboard:
            print('Logging to Tensorboard')
        if use_visdom:
            print('Logging to visdom')

    
    # Save model occasionally 
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX,step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
torch.save(net, 'wsddn_model.pt')
