def shuffle_list(*ls):
    seed(0)
    l =list(zip(*ls))

    shuffle(l)
    return zip(*l)
