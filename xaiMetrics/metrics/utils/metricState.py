from pip._internal.operations import freeze

class metricState:
    def __init__(self, name, version, language_models = None):
        self.name = name
        self.version = version
        self.modules = list(freeze.freeze())
        self.language_models = language_models

    def __str__(self):
        if self.language_models:
            return "-".join([self.name, self.version,self.language_models,"-".join(self.modules)])
        else:
            return "-".join([self.name, self.version,"-".join(self.modules)])
