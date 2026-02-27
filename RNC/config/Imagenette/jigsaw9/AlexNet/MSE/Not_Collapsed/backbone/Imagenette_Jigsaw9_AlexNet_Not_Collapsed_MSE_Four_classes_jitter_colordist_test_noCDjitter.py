batch_size   = 128

config = {}
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'Imagenette'
data_train_opt['split'] = 'train'
data_train_opt['pretext_mode'] = 'jigsaw_9'
data_train_opt['patch_jitter'] = 2
data_train_opt['color_distort'] = True
data_train_opt['color_dist_strength'] = 0.5

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'Imagenette'
data_test_opt['split'] = 'val'
data_test_opt['pretext_mode'] = 'jigsaw_9'
data_test_opt['patch_jitter'] = 0
data_test_opt['color_distort'] = False
data_test_opt['color_dist_strength'] = 0.0

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

net_opt = {}
net_opt['num_classes'] = 4

networks = {}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.01, 'momentum':0.9, 'weight_decay': 0, 'nesterov': True, 'LUT_lr': [(60, 0.01), (120, 0.002), (160, 0.0004), (200, 0.00008)]}
networks['model'] = {'def_file': 'architectures/AlexNet.py', 'pretrained': None, 'opt': net_opt, 'optim_params': net_optim_params}
config['networks'] = networks

config['nc_reg'] = {
    'layers':  ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'],
    'weights': {'conv1': 0.001,'conv2': 0.001,'conv3': 0.001,'conv4': 0.001,'conv5': 0.001,'fc6': 0.001,'fc7': 0.001,},
    'detach_sb': True
}


criterions = {}
criterions['loss'] = {'ctype':'MSELoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
