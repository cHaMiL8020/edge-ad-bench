DATASETS = {}
MODELS = {}

def register_dataset(name):
    def deco(cls):
        DATASETS[name] = cls
        return cls
    return deco

def register_model(name):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco
