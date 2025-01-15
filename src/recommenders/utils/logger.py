from torch.utils.tensorboard import SummaryWriter

class TrainingLogger(object):
    def __init__(self, tb_dir):
        self.tb_dir = tb_dir
        self.tb_writer = SummaryWriter(tb_dir)
   
    def add_scalar(self, name, value, step):
        self.tb_writer.add_scalar(name, value, step)

    def add_hparams(self, params, metrics):
        self.tb_writer.add_hparams(params, metrics)
        self.tb_writer.flush()
    
    # def add_text(self, name, value)
    #     self.tb_writer.add_text()
    
    def add_image(self, name,  image, step):
        self.tb_writer.add_image(name, image, step)
