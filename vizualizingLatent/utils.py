
import torch


def parse_data(training_data,TESTING_RATIO):
    ratio = TESTING_RATIO
    T = int(training_data.shape[0] * ratio)

    train_x = training_data[:-T]
    test_x = training_data[-T:]

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)

    return train_x, test_x


def tensor_to_numpy(array):

    return array.cpu().data.numpy()





