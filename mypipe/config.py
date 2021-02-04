class Config:
    def __init__(self, run_name, target="state", folds=5):
        self.RUN_NAME = run_name
        self.TARGET = target
        self.FOLDS = folds
        self.INPUT = '../input'
        self.OUTPUT = '../output'
        self.SUBMISSION = '../submission'
        self.NOTEBOOKS = '../notebooks'
        self.EXP = f"{self.OUTPUT}/{run_name}"
        self.PREDS = self.EXP + '/preds'
        self.COLS = self.EXP + '/cols'
        self.TRAINED = self.EXP + '/trained'
        self.FEATURE = self.EXP + '/feature'
        self.REPORTS = self.EXP + '/reports'
