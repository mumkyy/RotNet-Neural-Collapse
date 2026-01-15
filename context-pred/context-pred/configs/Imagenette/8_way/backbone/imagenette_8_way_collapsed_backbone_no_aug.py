batch_size  = 128

config = {}

data_train_opt = {} 
data_train_opt['dataset_name']   = 'Imagenette'
data_train_opt['dataset_root']   = 'data'
data_train_opt['split']          = 'train'
data_train_opt['batch_size']     = batch_size

# pretext-specific knobs
data_train_opt['patch_dim']      = 32          
data_train_opt['gap']            = 8           # set to 0 or None to "disable" gap
data_train_opt['mode']           = 'EIGHT'     # 'EIGHT' or 'QUAD'
data_train_opt['chromatic']      = False      
data_train_opt['jitter']         = False      

data_test_opt = {} 
data_test_opt['dataset_name']    = 'Imagenette'
data_test_opt['dataset_root']    = 'data'
data_test_opt['split']           = 'val'
data_test_opt['batch_size']      = batch_size

data_test_opt['patch_dim']       = data_train_opt['patch_dim']
data_test_opt['gap']             = data_train_opt['gap']
data_test_opt['mode']            = data_train_opt['mode']
data_test_opt['chromatic']       = data_train_opt['chromatic']
data_test_opt['jitter']          = data_train_opt['jitter']


config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

net_opt = {}

# IMPORTANT:
#   - EIGHT mode  -> 8 relative position classes
#   - QUAD mode   -> 4 classes (top/bottom, left/right, diagR, diagL)
net_opt['num_classes'] = 8      # change to 4 if mode='QUAD'
net_opt['patch_dim'] = data_train_opt['patch_dim']


networks = {}
net_optim_params = {
    'optim_type':   'adam',
    'lr':           0.0005,
    'weight_decay': 5e-4,
}
networks['model'] = {
    'def_file':     './model.py',
    'arch':         'AlexNetwork',   
    'opt':          net_opt,
    'optim_params': net_optim_params,
}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'MSELoss', 'opt': None}
config['criterions'] = criterions
