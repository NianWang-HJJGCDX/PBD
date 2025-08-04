
def create_model(opt):
    print(opt.model)
    if opt.model == 'PBD':
        assert(opt.dataset_mode == 'unaligned')
        from .PBD import PBD
        model = PBD()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single' or 'paired')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
