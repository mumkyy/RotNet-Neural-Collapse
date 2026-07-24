batch_size   = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar10'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar10'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 200

net_opt = {}
net_opt['num_classes'] = 4
net_opt['num_stages']  = 4
net_opt['use_avg_on_conv3'] = False

networks = {}
net_optim_params = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 0, 'nesterov': True, 'LUT_lr':[(60, 0.1),(120, 0.02),(160, 0.004),(200, 0.0008)]}
networks['model'] = {'def_file': 'architectures/NetworkInNetwork.py', 'pretrained': None, 'opt': net_opt,  'optim_params': net_optim_params}
config['networks'] = networks

# Layerwise NC-3 penalty (not collapsed: positive weights push the weight
# subspace away from the detached class-mean subspace, resisting NC3 collapse).
# No weight decay -- the class-mean matrix M is detached inside the penalty
# (stopgrad on features), so only the conv/classifier weights are driven.
# Exposed NIN feature keys: conv1, conv2, conv3, conv4, penult, classifier.
pabs_layers = [
    "conv2",
    "conv3",
    "conv4",
    "classifier",
]

pabs_weight = 1e-3

config["nc3_layerwise_pen"] = {
    "layers": pabs_layers,

    "weights": {
        "conv2": pabs_weight,
        "conv3": pabs_weight,
        "conv4": pabs_weight,
        "classifier": pabs_weight,
    },
    "no_svd": True,
}

criterions = {}
criterions['loss'] = {'ctype':'MSELoss', 'opt':None}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
