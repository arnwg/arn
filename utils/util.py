import matplotlib.pyplot as plt
import numpy as np
import torch

import pandas as pd
import os

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

from utils.plot import plot_pr_curve, plot_auc_curve

from data import data_loader
from PIL import Image

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        x = Image.fromarray(x.view(x.shape[1], x.shape[2]).numpy(), mode='L')

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def generate_labels(size, pflip, lb, ub, step, decay=.9995, up=True): # .9994536323918296

    if up:
        lb = ub - (ub-lb)*((decay)**step)
    else:
        ub = lb + (ub-lb)*((decay)**step)
    pflip = pflip*((decay)**step)

    y = np.random.uniform(lb, ub,size)

    sf = int(pflip*size)
    if sf > 0:
        y[:sf] = 1- y[:sf]
        np.random.shuffle(y)

    return torch.FloatTensor(y)


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, device, temperature):
    y = torch.log_softmax(logits, dim=-1) + sample_gumbel(logits.size(), device)
    return torch.softmax(y / temperature, dim=-1).to(device)


def gumbel_softmax(logits, device, temperature=1e-5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, device, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def gumbel_sigmoid_sample(logits, temperature):
    # See https://davidstutz.de/categorical-variational-auto-encoders-and-the-gumbel-trick/
    u = torch.rand_like(logits)
    # we exploit the fact log(sigma(x)) - log(1-sigma(x)) = x
    y = logits + torch.log(u) - torch.log(1 - u)

    return torch.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature=1e-5):
    """
    input: [*]
    return: [*] a binary response
    """
    y = gumbel_sigmoid_sample(logits, temperature)
    y_hard = (y > .5).float()
    return (y_hard - y).detach() + y


def predict(D, device, test_loader, is_mnist=False, thresh=.0):
    D.eval()
    i = 0

    for batch, label in test_loader:
        batch = batch.to(device)

        if is_mnist:
            batch = (batch > thresh).float()

        label = label.to(device)

        with torch.no_grad():
            y_pred = D(batch)

        if i == 0:
            y_true = label.cpu()
            yP = y_pred.cpu()
        else:
            y_true = torch.cat((y_true, label.cpu()))
            yP = torch.cat((yP, y_pred.cpu()))

        i += 1

    return y_true, yP

def save_arn_models(G, D, path_G, path_D):
    torch.save(D.state_dict(), path_D)
    torch.save(G.state_dict(), path_G)

def load_arn_models(G, D, path_G, path_D):
    if os.path.exists(path_G):
        G.load_state_dict(torch.load(path_G))

    if os.path.exists(path_D):
        D.load_state_dict(torch.load(path_D))


def save_gan_models(G, D, path_G, path_D):
    torch.save(D.state_dict(), path_D)
    torch.save(G.state_dict(), path_G)

def load_gan_models(G, D, path_G, path_D):
    if os.path.exists(path_G):
        G.load_state_dict(torch.load(path_G))

    if os.path.exists(path_D):
        D.load_state_dict(torch.load(path_D))

def get_data(params):
    DATASET_NAME = params['DATASET_NAME_ASS']

    if DATASET_NAME == 'KDDCUP99':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_KDDCUP99(params['PATH'], params['seed'], params['scale'],
                                            params['show'], params['is_reconstruction_score'], params['contamination'],
                                            params['percAnomalies'])
        else:
            return data_loader.get_KDDCUP99(params['PATH'], params['seed'], params['scale'],
                                        params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'KDDCUP99_Rev':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_KDDCUP99_REV(params['PATH'], params['seed'], params['scale'],
                                         params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])

        elif 'weak_supervision' in params and params['weak_supervision']:
            return data_loader.get_KDDCUP99_REV(params['PATH'], params['seed'],
                                                scale=params['scale'], show=params['show'],
                                                is_reconstruction_score=params['is_reconstruction_score'],
                                                percAnomalies=params['percAnomalies'],
                                                weak_supervision=params['weak_supervision'])

        else:
            return data_loader.get_KDDCUP99_REV(params['PATH'], params['seed'], params['scale'],
                                            params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'KDDCUP99_INV':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_KDDCUP99_INV(params['PATH'], params['seed'], params['scale'],
                                         params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_KDDCUP99_INV(params['PATH'], params['seed'], params['scale'],
                                            params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'NSL_KDD':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_NSLKDD(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                   params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_NSLKDD(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                      params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'DoH':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_DoH(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_DoH(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                   params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'DoH_INV':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_DoH_INV(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                    params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_DoH_INV(params['PATH'], params['PATH_2'], params['seed'], params['scale'],
                                       params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'CoverType':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_CoverType(params['PATH'], params['seed'], params['scale'],
                                      params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_CoverType(params['PATH'], params['seed'], params['scale'],
                                         params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'CreditCard':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_CreditCard(params['PATH'], params['seed'], params['scale'],
                                       params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])
        else:
            return data_loader.get_CreditCard(params['PATH'], params['seed'], params['scale'],
                                          params['show'], params['is_reconstruction_score'])
    elif DATASET_NAME == 'Bank':
        if 'contamination' in params and params['contamination']:
            return data_loader.get_Bank(params['PATH'], params['seed'], params['scale'],
                                 params['show'], params['is_reconstruction_score'], params['contamination'],
                                         params['percAnomalies'])

        elif 'weak_supervision' in params and params['weak_supervision']:
            return data_loader.get_Bank(params['PATH'], params['seed'],
                                                scale=params['scale'], show=params['show'],
                                                is_reconstruction_score=params['is_reconstruction_score'],
                                                percAnomalies=params['percAnomalies'],
                                                weak_supervision=params['weak_supervision'])

        else:
            return data_loader.get_Bank(params['PATH'], params['seed'], params['scale'],
                                    params['show'], params['is_reconstruction_score'])

    else:
        print('No data loader available')
        return


def get_Loader_weak(dataset, params):

    train_loader_normal = DataLoader(dataset=torch.FloatTensor(dataset['x_train'][dataset['y_train'] == 1]),
                                     batch_size = params['batch_size'], shuffle=True, drop_last = True)

    train_loader_abnormal = DataLoader(dataset=torch.FloatTensor(dataset['x_train'][dataset['y_train'] == 0]),
                                       batch_size = params['batch_size'], shuffle=True, drop_last = False)

    val_dataset = TensorDataset(torch.tensor(dataset['x_val']), torch.tensor(dataset['y_val']))
    test_dataset = TensorDataset(torch.tensor(dataset['x_test']), torch.tensor(dataset['y_test']))

    val_loader = DataLoader(dataset = val_dataset, batch_size=params['batch_size'], shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size=params['batch_size'], shuffle = False)

    return train_loader_normal, train_loader_abnormal, val_loader, test_loader


def get_Loader(dataset, params):

    if params['MODEL_NAME'] == 'FenceGAN' or params['MODEL_NAME'] == 'GANomaly' :
        train_ds = TensorDataset(torch.FloatTensor(dataset['x_train']),
                                 torch.FloatTensor(dataset['y_train']))
        train_loader = DataLoader(dataset = train_ds,
                                  batch_size=params['batch_size'], shuffle = True,
                                  drop_last = True)

    else:
        train_loader = DataLoader(dataset = torch.FloatTensor(dataset['x_train']),
                              batch_size=params['batch_size'], shuffle = True, drop_last = True)

    val_dataset = TensorDataset(torch.tensor(dataset['x_val']), torch.tensor(dataset['y_val']))
    test_dataset = TensorDataset(torch.tensor(dataset['x_test']), torch.tensor(dataset['y_test']))

    val_loader = DataLoader(dataset = val_dataset, batch_size=params['batch_size'], shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size=params['batch_size'], shuffle = False)

    return train_loader, val_loader, test_loader

def get_auprc(y_test, y_pred, show_pr_curve = True):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    auprc_score = auc(recall, precision)
    print(f'AUPRC: {auprc_score:.2f}')

    if show_pr_curve:
        plot_pr_curve(precision, recall)
    return auprc_score

def get_auc(y_test, y_pred, show_auc_curve = True):
    auc_score = roc_auc_score(y_test, y_pred)
    print(f'AUC: {auc_score:.2f}')

    if show_auc_curve:
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plot_auc_curve(fpr, tpr)

    return auc_score


def show_and_save_metrics(auc_list, pr_list, params):
    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    print('AUC:')
    print(auc_list)
    print('**************')
    print('AUPRC:')
    print(pr_list)
    print('**************')

    NAME = f'{MODEL_NAME}_{DATASET_NAME}'
    AUC_Frame = pd.DataFrame(auc_list, columns = [NAME])
    AUPRC_Frame = pd.DataFrame(pr_list, columns = [NAME])

    AU_NAME = f'{NAME}.csv'

    AUC_Frame.to_csv(os.path.join(params['SPACE_AUC'], AU_NAME), index=False)
    AUPRC_Frame.to_csv(os.path.join(params['SPACE_AUPRC'], AU_NAME), index=False)

    AUC_Frame = list(AUC_Frame[NAME])

    N = len(AUC_Frame)
    mean_auc = np.mean(AUC_Frame)
    std_auc = np.std(AUC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('AUC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')
    print('**************')

    AUPRC_Frame = list(AUPRC_Frame[NAME])

    N = len(AUPRC_Frame)
    mean_auc = np.mean(AUPRC_Frame)
    std_auc = np.std(AUPRC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('AUPRC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')

def runFenceGAN(params):
    from models.competitors.fencegan import FenceGAN

    SPACE_MODELS = params['SPACE_MODELS']
    SPACE_AUC = params['SPACE_AUC']
    SPACE_AUPRC = params['SPACE_AUPRC']

    for dir in [SPACE_MODELS, SPACE_AUC, SPACE_AUPRC]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    params['seed'] = 42

    dataset = get_data(params)

    nc = dataset['x_train'].shape[1]
    z_dim = 32
    gamma = 0.1
    alpha = 0.5
    beta = 30
    _power = 2
    v_freq = 4

    g_objective_anneal = 1
    repeat = 4
    baseline = 0.5

    n_runs = params['n_runs']
    auc_list = []
    n_epochs = params['num_epochs']
    auprc_list = []

    batch_size = params['batch_size']

    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    initial_seed = params['seed']
    device = params['device']
    print(f'Seed: {initial_seed}')

    for i in range(n_runs):
        print(f'Iteration: {i+1}')

        params['seed'] = initial_seed*(i+1)
        params['show'] = False

        dataset = get_data(params)

        name_G = f'{MODEL_NAME}_Generator_{DATASET_NAME}_{i}.ckpt'
        name_D = f'{MODEL_NAME}_Discriminator_{DATASET_NAME}_{i}.ckpt'

        path_G = os.path.join(params['SPACE_MODELS'], name_G)
        path_D = os.path.join(params['SPACE_MODELS'], name_D)

        train_loader, val_loader, test_loader = get_Loader(dataset, params)

        trainer = FenceGAN(nc, z_dim, gamma, alpha, beta, _power, v_freq, g_objective_anneal, repeat, baseline, device)

        _, _ , _= trainer.train(train_loader, test_loader, val_loader, path_G,
                             path_D, batch_size, n_epochs)

        load_gan_models(trainer.G, trainer.D, path_G, path_D)

        y_true, y_pred = trainer.predict(trainer.D, test_loader)

        y_true = 1-y_true
        y_pred = 1-y_pred

        auc_score = get_auc(y_true, y_pred)
        auprc_score = get_auprc(y_true, y_pred)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)
    show_and_save_metrics(auc_list, auprc_list, params)


def runGANomaly(params):
    from models.competitors.ganomaly import GANomaly

    SPACE_MODELS = params['SPACE_MODELS']
    SPACE_AUC = params['SPACE_AUC']
    SPACE_AUPRC = params['SPACE_AUPRC']

    for dir in [SPACE_MODELS, SPACE_AUC, SPACE_AUPRC]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    params['seed'] = 42
    dataset = get_data(params)

    args = {}
    nc = dataset['x_train'].shape[1]
    nz = 100
    manualseed = -1
    print_freq = 100
    args['w_adv'] = 1
    args['w_con'] = 50
    args['w_enc'] = 1
    isTrain = True

    n_runs = params['n_runs']
    auc_list = []
    n_epochs = params['num_epochs']
    auprc_list = []

    batch_size = params['batch_size']

    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    initial_seed = params['seed']
    device = params['device']
    print(f'Seed: {initial_seed}')

    for i in range(n_runs):
        print(f'Iteration: {i+1}')

        params['seed'] = initial_seed*(i+1)
        params['show'] = False

        dataset = get_data(params)

        name_G = f'{MODEL_NAME}_Generator_{DATASET_NAME}_{i}.ckpt'
        name_D = f'{MODEL_NAME}_Discriminator_{DATASET_NAME}_{i}.ckpt'

        path_G = os.path.join(params['SPACE_MODELS'], name_G)
        path_D = os.path.join(params['SPACE_MODELS'], name_D)

        train_loader, val_loader, test_loader = get_Loader(dataset, params)

        ganomaly = GANomaly(nc, nz, device)

        _, _, _ = ganomaly.train(train_loader, val_loader, args, path_G, path_D, n_epochs, batch_size)

        load_gan_models(ganomaly.G, ganomaly.D, path_G, path_D)
        labels, scores = ganomaly.test(ganomaly.G, test_loader)

        auc_score = get_auc(labels, scores)
        auprc_score = get_auprc(labels, scores)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)
    show_and_save_metrics(auc_list, auprc_list, params)


def runBaseline(params):
    if params['DATASET_NAME'] == 'CreditCard':
        from models.competitors.baseline_n import BaselineTrainer
    else:
        from models.competitors.baseline import BaselineTrainer

    SPACE_MODELS = params['SPACE_MODELS']
    SPACE_AUC = params['SPACE_AUC']
    SPACE_AUPRC = params['SPACE_AUPRC']

    for dir in [SPACE_MODELS, SPACE_AUC, SPACE_AUPRC]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    params['seed'] = 42
    dataset = get_data(params)

    params['nc'] = dataset['x_train'].shape[1]

    if 'discreteCol' in dataset:
        params['discreteCol'] = dataset['discreteCol']

    if 'selectedColumns' in dataset:
        params['selected_columns'] = dataset['selectedColumns']

    if 'index' in dataset:
        params['index'] = dataset['index']

    n_runs = params['n_runs']
    auc_list = []
    n_epochs = params['num_epochs']
    auprc_list = []

    batch_size = params['batch_size']

    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    initial_seed = params['seed']

    print(f'Seed: {initial_seed}')

    for i in range(n_runs):
        print(f'Iteration: {i+1}')

        params['seed'] = initial_seed*(i+1)
        params['show'] = False

        dataset = get_data(params)

        name = f'{MODEL_NAME}_{DATASET_NAME}_{i}.ckpt'
        path = os.path.join(params['SPACE_MODELS'], name)

        train_loader, val_loader, test_loader = get_Loader(dataset, params)

        baseline = BaselineTrainer(params)

        _, _ = baseline.train(train_loader, val_loader, path, batch_size, n_epochs)

        model_weights = torch.load(path)
        baseline.AE.load_state_dict(model_weights)
        labels, scores = baseline.predict(baseline.AE, test_loader)

        auc_score = get_auc(labels, scores)
        auprc_score = get_auprc(labels, scores)

        auc_list.append(auc_score)
        auprc_list.append(auprc_score)
    show_and_save_metrics(auc_list, auprc_list, params)


def show_metrics(params):
    DATASET_NAME = params['DATASET_NAME']
    MODEL_NAME = params['MODEL_NAME']

    NAME = f'{MODEL_NAME}_{DATASET_NAME}'
    AU_NAME = f'{NAME}.csv'

    AUC_Frame = pd.read_csv(os.path.join(params['SPACE_AUC'], AU_NAME))
    AUPRC_Frame = pd.read_csv(os.path.join(params['SPACE_AUPRC'], AU_NAME))

    AUC_Frame = list(AUC_Frame[NAME])

    N = len(AUC_Frame)
    mean_auc = np.mean(AUC_Frame)
    std_auc = np.std(AUC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('AUC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')
    print('**************')

    AUPRC_Frame = list(AUPRC_Frame[NAME])

    N = len(AUPRC_Frame)
    mean_auc = np.mean(AUPRC_Frame)
    std_auc = np.std(AUPRC_Frame)
    std_error = std_auc / (np.sqrt(N))

    ci = 1.96 * std_error
    lower_bound = mean_auc - ci
    upper_bound = mean_auc + ci

    print('AUPRC')
    print(f'{mean_auc:.2f} +/- {ci:.2f}')
    print(f'95% confidence level, average auc would be between {lower_bound:.2f} and {upper_bound:.2f}')




