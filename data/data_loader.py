### IMPORT
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def _resize(imageSet):
    import cv2
    tmp = []

    for i in range(imageSet.shape[0]):
        img = imageSet[i]
        resized_img = cv2.copyMakeBorder(img, 0, 0, 0, 0, borderType = cv2.BORDER_CONSTANT, value=[0])

        resized_img = cv2.resize(resized_img, (32, 32))

        tmp.append(resized_img)

    return np.array(tmp)


def _set_class_index(is_reconstruction_score = False):
    '''
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: index of anomalous and normal class
    '''
    if is_reconstruction_score:
        anoIndex = 1
        normIndex = 0
    else:
        anoIndex = 0
        normIndex = 1
    return anoIndex, normIndex


def _encoding(df, name):
    """
    One hot encoding for categorical attributes (i.e. [1, 0, 0], [0, 1, 0], [0, 0, 1] for A, B, C)
    :param df: dataframe
    :param name: name of the categorical attribute
    :return:
        names: the list of the new column names (i.e. [name_A, name_B, name_C])
        tmpL: the list of the mapping attribute - column position (i.e. [[0, A], [1, B], [2, C]])
    """
    names = []
    dummies = pd.get_dummies(df.loc[:,name])
    i = 0

    tmpL = []
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
        names.append(dummy_name)
        _x = [i, x]
        tmpL.append(_x)
        i += 1

    df.drop(name, axis=1, inplace=True)
    return names, tmpL


def _to_xy(df, target):
    '''
    Converts dataframe into x,y inputs
    :param df: dataframe
    :param target: class label
    :return:
        x: data
        y: class label
    '''
    y = df[target]
    x = df.drop(columns=target)
    return x, y

def removeB(df):
    """
    :param df: dataframe
    :return:
    """
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()

    for col in str_df:
        df[col] = str_df[col]

    return df

def get_Bank(PATH, seed, scale, show, is_reconstruction_score, contamination = False,
             percAnomalies=.0, weak_supervision = False):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    from scipy.io import arff

    data, _ = arff.loadarff(PATH)
    df = pd.DataFrame(data)
    df = removeB(df)

    discreteCol = df[df.columns.difference(['y'])].columns.tolist()
    columns = df.columns

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == no: normal class; class == yes: anomalous class
    labels = df['y'].copy()
    labels[labels != 'no'] = anoIndex # anomalous
    labels[labels == 'no'] = normIndex # normal

    df['y'] = labels
    normal = df[df['y'] == normIndex]
    abnormal = df[df['y'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    train_size = 26383
    val_size = 2551

    if weak_supervision:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(abnormal_2)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
    elif contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['y'].replace({anoIndex:normIndex}, inplace = True)
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = 580
    test_size = 1740
    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='y')
    x_val, y_val = _to_xy(val_set, target='y')
    x_test, y_test = _to_xy(test_set, target='y')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []
        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['discreteCol'] = discreteCol
    dataset['selectedColumns'] = selected_columns
    dataset['index'] = index

    return dataset


def get_KDDCUP99(PATH, seed, scale, show, is_reconstruction_score, contamination = False, percAnomalies=.0):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label']

    df = pd.read_csv(PATH, header=None, names=columns)
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    labels = df['label'].copy()
    # Class == normal: normal class; class != normal: anomalous class
    labels[labels != 'normal.'] = anoIndex # anomalous
    labels[labels == 'normal.'] = normIndex # normal

    df['label'] = labels
    normal = df[df['label'] == normIndex]
    abnormal = df[df['label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    train_size = int(len(normal)*.8)
    val_size = int(len(normal)*.05)+1

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['label'].replace({anoIndex:normIndex}, inplace = True)
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = int(len(abnormal_1)*.05)+1
    test_size = int(len(abnormal_1)*.15)+1

    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset


def get_KDDCUP99_INV(PATH, seed, scale, show, is_reconstruction_score, contamination = False, percAnomalies=.0):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label']

    df = pd.read_csv(PATH, header=None, names=columns)
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == normal: anomalous class; class != normal: normal class
    labels = df['label'].copy()
    labels[labels != 'normal.'] = normIndex # normal
    labels[labels == 'normal.'] = anoIndex # anomalous

    df['label'] = labels
    normal = df[df['label'] == normIndex]
    abnormal = df[df['label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))+1

    train_size = 274006
    val_size = 31540

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['label'].replace({anoIndex:normIndex}, inplace = True)
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_abnormal = abnormal_1[:val_size_ab]
    test_abnormal = abnormal_1[val_size_ab:val_size_ab+test_size_ab]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset


def get_DoH(PATH_B, PATH_M, seed, scale, show, is_reconstruction_score, contamination = False, percAnomalies=.0):
    '''
    :param PATH_B: the path of the benign dataset
    :param PATH_M: the path of the malicious dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
               'TimeStamp', 'Duration', 'FlowBytesSent', 'FlowSentRate',
               'FlowBytesReceived', 'FlowReceivedRate', 'PacketLengthVariance',
               'PacketLengthStandardDeviation', 'PacketLengthMean',
               'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
               'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation',
               'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean',
               'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian',
               'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation',
               'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
               'ResponseTimeTimeMean', 'ResponseTimeTimeMedian',
               'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian',
               'ResponseTimeTimeSkewFromMode',
               'ResponseTimeTimeCoefficientofVariation', 'Label']

    benign = pd.read_csv(PATH_B)
    malicious = pd.read_csv(PATH_M)

    df = pd.concat([benign, malicious])
    df.drop(columns=['TimeStamp', 'ResponseTimeTimeMedian', 'ResponseTimeTimeSkewFromMedian'], inplace = True)
    discreteCol = ['SourceIP', 'DestinationIP']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == Benign: normal class, class != benign: anomalous class
    labels = df['Label'].copy()
    labels[labels != 'Benign'] = anoIndex # anomalous
    labels[labels == 'Benign'] = normIndex # normal

    df['Label'] = labels
    normal = df[df['Label'] == normIndex]
    abnormal = df[df['Label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    train_size = int(len(normal)*.8)+1
    val_size = int(len(normal)*.05)

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['Label'].replace({anoIndex:normIndex}, inplace = True)

    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = int(len(abnormal_1)*.05)+1
    test_size = int(len(abnormal_1)*.15)+1
    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='Label')
    x_val, y_val = _to_xy(val_set, target='Label')
    x_test, y_test = _to_xy(test_set, target='Label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset

def get_DoH_INV(PATH_B, PATH_M, seed, scale, show, is_reconstruction_score, contamination = False,
                percAnomalies=.0):
    '''
    :param PATH_B: the path of the benign dataset
    :param PATH_M: the path of the malicious dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
               'TimeStamp', 'Duration', 'FlowBytesSent', 'FlowSentRate',
               'FlowBytesReceived', 'FlowReceivedRate', 'PacketLengthVariance',
               'PacketLengthStandardDeviation', 'PacketLengthMean',
               'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
               'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation',
               'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean',
               'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian',
               'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation',
               'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
               'ResponseTimeTimeMean', 'ResponseTimeTimeMedian',
               'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian',
               'ResponseTimeTimeSkewFromMode',
               'ResponseTimeTimeCoefficientofVariation', 'Label']

    benign = pd.read_csv(PATH_B)
    malicious = pd.read_csv(PATH_M)

    df = pd.concat([benign, malicious])
    df.drop(columns=['TimeStamp', 'ResponseTimeTimeMedian', 'ResponseTimeTimeSkewFromMedian'], inplace = True)
    discreteCol = ['SourceIP', 'DestinationIP']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == Benign: anomalous class, class != Benign: normal class
    labels = df['Label'].copy()
    labels[labels != 'Benign'] = normIndex # normal
    labels[labels == 'Benign'] = anoIndex # anomalous

    df['Label'] = labels
    normal = df[df['Label'] == normIndex]
    abnormal = df[df['Label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    train_size = 184444
    val_size = 15598

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['Label'].replace({anoIndex:normIndex}, inplace = True)
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = 2475
    test_size = 7427
    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='Label')
    x_val, y_val = _to_xy(val_set, target='Label')
    x_test, y_test = _to_xy(test_set, target='Label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset


def get_CoverType(PATH, seed, scale, show, is_reconstruction_score, contamination = False, percAnomalies=.0):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    df = pd.read_csv(PATH, header = None)
    discreteCol = np.arange(10, len(df.columns)-1)

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class in [1, 2, 3]: normal class, Class in [4, 5, 6, 7]: anomalous class
    labels = df[54].copy()
    labels.replace({1: normIndex, 2: normIndex, 3: normIndex,
                    4:anoIndex, 5:anoIndex, 6:anoIndex, 7:anoIndex}, inplace = True)
    df[54] = labels

    normal = df[df[54] == normIndex]
    abnormal = df[df[54] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))+1

    test_size_n = int(.15 * (len(normal) + len(abnormal_1)) - test_size_ab)
    val_size_n = int(.05 * (len(normal) + len(abnormal_1)) - val_size_ab)

    train_size = int(len(normal) - val_size_n - test_size_n)

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set[54].replace({anoIndex:normIndex}, inplace = True)
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal_1[:val_size_ab]
    test_abnormal = abnormal_1[val_size_ab:val_size_ab+test_size_ab]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target=54)
    x_val, y_val = _to_xy(val_set, target=54)
    x_test, y_test = _to_xy(test_set, target=54)

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        selected_columns[name] = x_train.columns.get_loc(name)

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(df.columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset


def get_KDDCUP99_REV(PATH, seed, scale, show, is_reconstruction_score, contamination = False, percAnomalies=.0,
                     weak_supervision = False):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label']

    df = pd.read_csv(PATH, header=None, names=columns)
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    # Delete nepture, smurf
    df_neptune = df[df['label'] == 'neptune.']
    df_smurf = df[df['label'] == 'smurf.']
    df = df.loc[~df.index.isin(df_neptune.index)]
    df = df.loc[~df.index.isin(df_smurf.index)]

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == normal: normal class; class != normal: anomalous class
    labels = df['label'].copy()
    labels[labels != 'normal.'] = anoIndex # anomalous
    labels[labels == 'normal.'] = normIndex # normal

    df['label'] = labels
    normal = df[df['label'] == normIndex]
    abnormal = df[df['label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))

    test_size_n = int(.15 * (len(normal) + len(abnormal_1)) - test_size_ab + 1)
    val_size_n = int(.05 * (len(normal) + len(abnormal_1)) - val_size_ab + 1)

    train_size = int(len(normal) - val_size_n - test_size_n)+1

    if contamination:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['label'].replace({anoIndex:normIndex}, inplace = True)
    elif weak_supervision:
        train_normal = normal[:train_size]
        train_abnormal = abnormal_2[:int(len(abnormal_2)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
    else:
        train_set = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal[:val_size_ab]
    test_abnormal = abnormal[val_size_ab:val_size_ab+test_size_ab]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset

def get_CreditCard(PATH, seed, scale, show, is_reconstruction_score):
    '''
    :param PATH: the path of the dataset
    :param seed: seed
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    df = pd.read_csv(PATH)

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == 0: normal class; class == 1: anomalous class
    labels = df['Class'].copy()

    if not is_reconstruction_score:
        labels.replace({0: normIndex, 1: anoIndex}, inplace = True)

    df['Class'] = labels

    normal = df[df['Class'] == normIndex]
    abnormal = df[df['Class'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    test_size_ab = int(len(abnormal)*(3/4))
    val_size_ab = int(len(abnormal)*(1/4))

    test_size_n = int(.15 * (len(normal) + len(abnormal)) - test_size_ab)
    val_size_n = int(.05 * (len(normal) + len(abnormal)) - val_size_ab)

    train_size = int(len(normal) - val_size_n - test_size_n)

    train_set = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal[:val_size_ab]
    test_abnormal = abnormal[val_size_ab:val_size_ab+test_size_ab]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='Class')
    x_val, y_val = _to_xy(val_set, target='Class')
    x_test, y_test = _to_xy(test_set, target='Class')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['scaler'] = scaler

    return dataset



def get_NSLKDD(PATH_TRAIN, PATH_TEST, seed, scale, show, is_reconstruction_score,
               contamination = False, percAnomalies=.0,
               mx = 0.889, mz = 0.028, my = 0.083):
    '''
    :param PATH_TRAIN: the path of the original training dataset
    :param PATH_TEST: the path of the original test dataset
    :param seed: seed
    :param mx: ratio for training set
    :param mz: ratio for validation set
    :param my: ratio for test set
    :param scale: normalize or not
    :param show: show the statistics
    :param is_reconstruction_score: reconstruction score or discriminator score
    :return: dataset (dict): containing the data and information about categorical attributes
    '''

    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_hot_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate','dst_host_srv_rerror_rate', 'label', 'unknown']

    train = pd.read_csv(PATH_TRAIN, delimiter = ',', header = None, names = columns)
    test = pd.read_csv(PATH_TEST, delimiter = ',', header = None, names = columns)

    train.drop(columns = ['unknown'], inplace = True)
    test.drop(columns = ['unknown'], inplace = True)

    rest = set(train.columns) - set(test.columns)
    for i in rest:
        idx = train.columns.get_loc(i)
        test.insert(loc=idx, column=i, value=0)


    df = pd.concat((train, test))
    discreteCol = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_hot_login', 'is_guest_login']

    names = []
    oneHot = dict()
    for name in discreteCol:
        n, t = _encoding(df, name)
        names.extend(n)
        oneHot[name] = t

    anoIndex, normIndex = _set_class_index(is_reconstruction_score)

    # Class == normal: normal class; class != normal: anomalous class
    labels = df['label'].copy()
    labels[labels != 'normal'] = anoIndex # anomalous
    labels[labels == 'normal'] = normIndex # normal

    df['label'] = labels
    normal = df[df['label'] == normIndex]
    abnormal = df[df['label'] == anoIndex]

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    if contamination:
        train_normal = normal[:int(mx*len(normal))]
        train_abnormal = abnormal_2[:int(len(train_normal)*percAnomalies)]
        train_set = pd.concat((train_normal, train_abnormal))
        train_set['label'].replace({anoIndex:normIndex}, inplace = True)

    else:
        train_set = normal[:int(mx*len(normal))]

    val_normal = normal[int(mx*len(normal)): int(mx*len(normal))+int(mz*len(normal))]
    test_normal = normal[int(mx*len(normal))+int(mz*len(normal)): ]

    val_abnormal = abnormal_1[:int(mz*len(normal))]
    test_abnormal = abnormal_1[int(mz*len(normal)):int(mz*len(normal))+int(my*len(normal))+1]

    val_set = pd.concat((val_normal, val_abnormal))
    test_set = pd.concat((test_normal, test_abnormal))

    x_train, y_train = _to_xy(train_set, target='label')
    x_val, y_val = _to_xy(val_set, target='label')
    x_test, y_test = _to_xy(test_set, target='label')

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == normIndex])} normal records and {len(x_train[y_train == anoIndex])} abnormal records')

        if contamination:
            print(f'There are {len(train_abnormal)} records anomalies labeled as normal')

        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == normIndex])} normal records and {len(x_val[y_val == anoIndex])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == normIndex])} normal records and {len(x_test[y_test == anoIndex])} abnormal records')

    selected_columns = dict()

    for name in discreteCol:
        cols = [col for col in names if name in col]
        tmp = []

        for c in cols:
            tmp.append(x_train.columns.get_loc(c))

        selected_columns[name] = tmp

    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    x_test = x_test.to_numpy()

    index = np.arange(0, len(columns)-len(discreteCol)-1)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, index])
        x_train[:, index] = scaler.transform(x_train[:, index])
        x_val[:, index] = scaler.transform(x_val[:, index])
        x_test[:, index] = scaler.transform(x_test[:, index])

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['selectedColumns'] = selected_columns
    dataset['discreteCol'] = discreteCol
    dataset['oneHot'] = oneHot
    dataset['index'] = index
    dataset['scaler'] = scaler

    return dataset

def getMNIST(idx_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    if is_arn:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

    else:
        transform = transforms.Compose([transforms.Resize(32),
            transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets == idx_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets == idx_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets != idx_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets != idx_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    train_size = int(len(normal)*.8)
    val_size = int(len(normal)*.05)+1

    x_train = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = int(len(abnormal_1)*.05)+1
    test_size = int(len(abnormal_1)*.15)+1

    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    x_val = np.concatenate((val_normal, val_abnormal))
    x_test = np.concatenate((test_normal, test_abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_val = np.concatenate((np.ones(val_normal.shape[0]),
                                np.zeros(val_abnormal.shape[0])))
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(test_abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_val = np.concatenate((np.zeros(val_normal.shape[0]),
                                np.ones(val_abnormal.shape[0])))
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(test_abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == idx_norm])} normal records and {len(x_val[y_val == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    new_x_train = np.copy(x_train)
    new_x_val = np.copy(x_val)
    new_x_test = np.copy(x_test)

    dataset['x_train'] = new_x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = new_x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = new_x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_val'] = np.expand_dims(dataset['x_val'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset, transform



def getMNIST_REV(idx_anl_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    # To normalize in range [-1, 1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if is_arn:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor()])

    else:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        ])
        #transforms.Normalize((0.1307,), (0.3081,))

    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets != idx_anl_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets != idx_anl_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets == idx_anl_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets == idx_anl_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))

    test_size_n = int(.15 * (len(normal) + len(abnormal_1)) - test_size_ab + 1)
    val_size_n = int(.05 * (len(normal) + len(abnormal_1)) - val_size_ab + 1)

    train_size = int(len(normal) - val_size_n - test_size_n)+1
    x_train = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal[:val_size_ab]
    test_abnormal = abnormal[val_size_ab:val_size_ab+test_size_ab]

    x_val = np.concatenate((val_normal, val_abnormal))
    x_test = np.concatenate((test_normal, test_abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_val = np.concatenate((np.ones(val_normal.shape[0]),
                                np.zeros(val_abnormal.shape[0])))
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(test_abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_val = np.concatenate((np.zeros(val_normal.shape[0]),
                                np.ones(val_abnormal.shape[0])))
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(test_abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == idx_norm])} normal records and {len(x_val[y_val == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_val'] = np.expand_dims(dataset['x_val'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset, transform

def getMNIST_Normal(idx_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    if is_arn:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor()])

    else:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets == idx_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets == idx_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets != idx_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets != idx_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    train_size = int(len(normal)*.8)
    val_size = int(len(normal)*.05)+1

    x_train = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = int(len(abnormal_1)*.05)+1
    test_size = int(len(abnormal_1)*.15)+1

    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    x_val = np.concatenate((val_normal, val_abnormal))
    x_test = np.concatenate((test_normal, test_abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_val = np.concatenate((np.ones(val_normal.shape[0]),
                                np.zeros(val_abnormal.shape[0])))
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(test_abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_val = np.concatenate((np.zeros(val_normal.shape[0]),
                                np.ones(val_abnormal.shape[0])))
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(test_abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == idx_norm])} normal records and {len(x_val[y_val == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_val'] = np.expand_dims(dataset['x_val'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset, transform

def getMNIST_Rev_Res(idx_anl_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    # To normalize in range [-1, 1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if is_arn:
        transform = transforms.Compose([transforms.ToTensor()])

    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])


    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets != idx_anl_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets != idx_anl_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets == idx_anl_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets == idx_anl_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    if is_arn:
        normal = (normal - np.min(normal)) / (np.max(normal) - np.min(normal))
        abnormal = (abnormal - np.min(abnormal)) / (np.max(abnormal) - np.min(abnormal))

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)]
    abnormal_2 = abnormal[int(len(abnormal)*.5):]

    test_size_ab = int(len(abnormal_1)*(3/4))
    val_size_ab = int(len(abnormal_1)*(1/4))

    test_size_n = int(.15 * (len(normal) + len(abnormal_1)) - test_size_ab + 1)
    val_size_n = int(.05 * (len(normal) + len(abnormal_1)) - val_size_ab + 1)

    train_size = int(len(normal) - val_size_n - test_size_n)+1
    x_train = normal[:train_size]

    val_normal = normal[train_size: train_size+val_size_n]
    test_normal = normal[train_size+val_size_n: ]

    val_abnormal = abnormal[:val_size_ab]
    test_abnormal = abnormal[val_size_ab:val_size_ab+test_size_ab]

    x_val = np.concatenate((val_normal, val_abnormal))
    x_test = np.concatenate((test_normal, test_abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_val = np.concatenate((np.ones(val_normal.shape[0]),
                                np.zeros(val_abnormal.shape[0])))
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(test_abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_val = np.concatenate((np.zeros(val_normal.shape[0]),
                                np.ones(val_abnormal.shape[0])))
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(test_abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == idx_norm])} normal records and {len(x_val[y_val == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    if not is_arn:
        dataset['x_train'] = _resize(dataset['x_train'])
        dataset['x_val'] = _resize(dataset['x_val'])
        dataset['x_test'] = _resize(dataset['x_test'])

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_val'] = np.expand_dims(dataset['x_val'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset


def getMNIST_Res(idx_nrl_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    # To normalize in range [-1, 1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if is_arn:
        transform = transforms.Compose([transforms.ToTensor()])

    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])


    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets == idx_nrl_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets == idx_nrl_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets != idx_nrl_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets != idx_nrl_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    if is_arn:
        normal = (normal - np.min(normal)) / (np.max(normal) - np.min(normal))
        abnormal = (abnormal - np.min(abnormal)) / (np.max(abnormal) - np.min(abnormal))

    normal = shuffle(normal, random_state = seed)
    abnormal = shuffle(abnormal, random_state = seed)

    abnormal_1 = abnormal[:int(len(abnormal)*.5)+1]
    abnormal_2 = abnormal[int(len(abnormal)*.5)+1:]

    train_size = int(len(normal)*.8)
    val_size = int(len(normal)*.05)+1

    x_train = normal[:train_size]
    val_normal = normal[train_size: train_size+val_size]
    test_normal = normal[train_size+val_size: ]

    val_size = int(len(abnormal_1)*.05)+1
    test_size = int(len(abnormal_1)*.15)+1

    val_abnormal = abnormal_1[:val_size]
    test_abnormal = abnormal_1[val_size:val_size+test_size]

    x_val = np.concatenate((val_normal, val_abnormal))
    x_test = np.concatenate((test_normal, test_abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_val = np.concatenate((np.ones(val_normal.shape[0]),
                                np.zeros(val_abnormal.shape[0])))
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(test_abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_val = np.concatenate((np.zeros(val_normal.shape[0]),
                                np.ones(val_abnormal.shape[0])))
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(test_abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print(f'We use {len(abnormal_1)} anomalous records')
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_val)} records in validation set')
        print(f'Validation set is composed by {len(x_val[y_val == idx_norm])} normal records and {len(x_val[y_val == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_val'] = x_val.astype(np.float32)
    dataset['y_val'] = y_val.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    if not is_arn:
        dataset['x_train'] = _resize(dataset['x_train'])
        dataset['x_val'] = _resize(dataset['x_val'])
        dataset['x_test'] = _resize(dataset['x_test'])

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_val'] = np.expand_dims(dataset['x_val'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset


def getMNIST_REV_GanomalyProt(idx_anl_class, is_arn = True, seed = 123, show =True):
    np.random.seed(seed)

    # To normalize in range [-1, 1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if is_arn:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor()])

    else:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

    normal_1 = (train_set.data[np.where(train_set.targets != idx_anl_class)]).numpy()
    normal_2 = (test_set.data[np.where(test_set.targets != idx_anl_class)]).numpy()

    abnormal_1 = (train_set.data[np.where(train_set.targets == idx_anl_class)]).numpy()
    abnormal_2 = (test_set.data[np.where(test_set.targets == idx_anl_class)]).numpy()

    normal = np.concatenate((normal_1, normal_2))
    abnormal = np.concatenate((abnormal_1, abnormal_2))

    # Split the normal data into the new train and tests.
    idx = np.arange(len(normal))
    np.random.seed(seed)
    np.random.shuffle(idx)

    nrm_trn_len = int(len(idx) * 0.80)
    nrm_trn_idx = idx[:nrm_trn_len]
    nrm_tst_idx = idx[nrm_trn_len:]

    x_train = normal[nrm_trn_idx]
    test_normal = normal[nrm_tst_idx]

    x_test = np.concatenate((test_normal, abnormal))

    if is_arn:
        # 1: Normal, 0: Anomalous
        idx_norm = 1
        idx_anomal = 0
        y_train = np.ones(x_train.shape[0])
        y_test = np.concatenate((np.ones(test_normal.shape[0]),
                                 np.zeros(abnormal.shape[0])))

    else:
        # 0: Normal, 1: Anomalous
        idx_norm = 0
        idx_anomal =  1
        y_train = np.zeros(x_train.shape[0])
        y_test = np.concatenate((np.zeros(test_normal.shape[0]),
                                 np.ones(abnormal.shape[0])))

    if show:
        print('{} normal records, {} anormal records'.format(len(normal), len(abnormal)))
        print('-' * 89)
        print(f'There are {len(x_train)} records in training set')
        print(f'Training set is composed by {len(x_train[y_train == idx_norm])} normal records and {len(x_train[y_train == idx_anomal])} abnormal records')
        print('-' * 89)
        print(f'There are {len(x_test)} records in test set')
        print(f'Test set is composed by {len(x_test[y_test == idx_norm])} normal records and {len(x_test[y_test == idx_anomal])} abnormal records')

    dataset = {}

    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)

    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    dataset['x_train'] = np.expand_dims(dataset['x_train'], axis=1)
    dataset['x_test'] = np.expand_dims(dataset['x_test'], axis=1)

    return dataset, transform





