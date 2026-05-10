def get_model(model_name, args):
    name = model_name.lower()
    if name=='area':
        from models.area import Learner
        return Learner(args)
    else:
        assert 0
