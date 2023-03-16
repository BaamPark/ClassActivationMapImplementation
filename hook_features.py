#SaveFeatures instance use pytorch hook function
#forward hook allows model to execute usr defined command for every forward pass of the input
class SaveFeatures():
    features=None
    def __init__(self, m):
        self.hook =  m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = ((output.cpu()).data).numpy()
    def remove(self): 
        self.hook.remove()