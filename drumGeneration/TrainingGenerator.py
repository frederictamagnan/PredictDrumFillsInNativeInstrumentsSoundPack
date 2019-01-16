class TrainingGenerator:




    def __init__(self):

        self.dataset_filepath='/home/ftamagna/Documents/_AcademiaSinica/dataset/drumGeneration/fills_reduced.npz'

    def load_data(self):
        self.data = np.load(self.dataset_filepath)

    def train(self):
        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)