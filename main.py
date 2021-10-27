import json
import os
import sys
import numpy as np
import torch

from utils.util import load_arn_models, get_data, get_Loader, get_auc, get_auprc, show_and_save_metrics, predict
from utils.util import get_Loader_weak
def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def run(params):
    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    n_runs = params['n_runs']

    if 'start_runs'  in params:
        start_runs = params['start_runs']
    else:
        start_runs = 0

    seed = params['seed']

    auc_list = []
    auprc_list = []

    for i in range(start_runs, n_runs):
        print(f'Iteration: {i}')
        params['seed'] = seed*(i+1)
        dataset = get_data(params)

        if 'weak_supervision' in params:
            train_loader_normal, train_loader_abnormal, val_loader, test_loader = get_Loader_weak(dataset, params)
        else:
            train_loader, val_loader, test_loader = get_Loader(dataset, params)

        idx = params['MODEL_PATH'].rfind('.')
        mod = __import__(params['MODEL_PATH'][:idx], fromlist=params['MODEL_PATH'][idx+1:])
        Model = getattr(mod, params['MODEL_PATH'][idx+1:])

        model = Model(params)

        name_G = f'{MODEL_NAME}_Generator_{DATASET_NAME}_{i}.ckpt'
        name_D = f'{MODEL_NAME}_Discriminator_{DATASET_NAME}_{i}.ckpt'

        path_G = os.path.join(params['SPACE_MODELS'], name_G)
        path_D = os.path.join(params['SPACE_MODELS'], name_D)

        ### Training ###
        if 'weak_supervision' in params:
            _ = model.train(train_loader_normal, train_loader_abnormal, val_loader, path_G, path_D,
                            batch_size=params['batch_size'], num_epochs=params['num_epochs'])
        else:
            _ = model.train(train_loader, val_loader, path_G, path_D, batch_size=params['batch_size'],
                            num_epochs=params['num_epochs'])

        ### Evaluation ###
        load_arn_models(model.G, model.D, path_G, path_D)
        y_true, y_pred = predict(model.D, params['device'], test_loader)

        y_true = 1-y_true
        y_pred = 1-y_pred

        auc_score = get_auc(y_true, y_pred)
        auprc_score = get_auprc(y_true, y_pred)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)
    show_and_save_metrics(auc_list, auprc_list, params)

def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = params['n_gpu']

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f'Device: {device}')

    DATASET_AREA = params['DATASET_AREA']

    FILE = params['FILE_NAME']
    params['PATH'] = os.path.join(DATASET_AREA, FILE)

    if 'FILE_NAME_2' in params:
        FILE_2 = params['FILE_NAME_2']
        params['PATH_2'] = os.path.join(DATASET_AREA, FILE_2)

    SPACE_MODELS = params['SPACE_MODELS']
    SPACE_AUC = params['SPACE_AUC']
    SPACE_AUPRC = params['SPACE_AUPRC']

    for dir in [SPACE_MODELS, SPACE_AUC, SPACE_AUPRC]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    params['device'] = device
    params['seed'] = 42

    dataset = get_data(params)

    params['show'] = False
    params['nc'] = dataset['x_train'].shape[1]

    if 'discreteCol' in dataset:
        params['discreteCol'] = dataset['discreteCol']

    if 'selectedColumns' in dataset:
        params['selected_columns'] = dataset['selectedColumns']

    if 'index' in dataset:
        params['index'] = dataset['index']

    run(params)

if __name__ == '__main__':
    main(sys.argv[1])


