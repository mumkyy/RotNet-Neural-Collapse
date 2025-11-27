batch_size  = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['dataset_name'] = 'Imagenette'
data_train_opt['dataset_root'] = 'data'
data_train_opt['split'] = 'train'
data_train_opt['batch_size'] = batch_size
data_train_opt['patch_dim'] = 32
data_train_opt['gap'] = 8


data_test_opt = {} 
data_test_opt['dataset_name'] = 'Imagenette'
data_test_opt['dataset_root'] = 'data'
data_test_opt['split'] = 'val'
data_test_opt['batch_size'] = batch_size
data_test_opt['patch_dim'] = data_train_opt['patch_dim']
data_test_opt['gap'] = data_train_opt['gap']


config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

net_opt = {}
net_opt['num_classes'] = 8

networks = {}
net_optim_params = {'optim_type': 'adam', 'lr': 0.0005, 'weight_decay': 5e-4}
networks['model'] = {'def_file': './model.py', 'arch': 'AlexNetwork',  'opt': net_opt,  'optim_params': net_optim_params} 
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'MSELoss', 'opt':None}
config['criterions'] = criterions

