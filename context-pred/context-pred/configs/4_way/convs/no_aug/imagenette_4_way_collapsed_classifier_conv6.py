batch_size = 128

config = {}

data_train_opt = {}
data_train_opt['batch_size']        = batch_size
data_train_opt['dataset_name']      = 'Imagenette'   
data_train_opt['dataset_root']   = 'data'
data_train_opt['split']             = 'train'

data_test_opt = {}
data_test_opt['batch_size']        = batch_size
data_test_opt['dataset_name']      = 'Imagenette'  
data_test_opt['dataset_root']   = 'data'
data_test_opt['split']             = 'val'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

config['max_num_epochs'] = 100

networks = {}

feat_pretrained_file = './checkpoints/4_way/backbone/no_aug/imagenette_4_way_collapsed_backbone/200.pt'

cls_net_opt = {
    'num_classes':   10,                     
    'backbone_ckpt': feat_pretrained_file,   
    'freeze_backbone': True,                 
    'head_feat_key': 'conv6',                  
    'input_size':    160,                     
}

cls_net_optim_params = {
    'optim_type': 'adam',
    'lr':         0.0005,
    'weight_decay': 5e-4,
}

networks['model'] = {
    'def_file':    './model.py',       
    'arch':        'AlexClassifier',   
    'opt':         cls_net_opt,
    'optim_params': cls_net_optim_params,
}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype': 'CrossEntropyLoss', 'opt': None}
config['criterions'] = criterions

