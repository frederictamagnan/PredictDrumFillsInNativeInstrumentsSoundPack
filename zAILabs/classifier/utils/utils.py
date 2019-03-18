def tensor_to_numpy(array):

    return array.cpu().data.numpy()