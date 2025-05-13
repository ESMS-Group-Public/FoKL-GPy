from src.FoKL.samplers.gibbs import Sampler1
from src.FoKL.samplers.gibbs_Xin_update import Sampler2


class samplers:
    def __init__(self, fokl, config, functions):
        self.fokl = fokl
        self.config = config
        self.functions = functions
        self.sampler1 = Sampler1(self.fokl, self.config, self.functions)
        self.sampler2 = Sampler2(self.fokl, self.config, self.functions)
        

    def run_sampler(self, sampler, *args, **kwargs):
        if sampler == 'gibbs':
            return self.sampler1.gibbs(*args, **kwargs)
        elif sampler == 'gibbs_Xin_update':
            return self.sampler2.gibbs_Xin_update(*args, **kwargs)
        else:
            raise ValueError("Unsupported sampler type selected.")