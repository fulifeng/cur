import copy
import fold_evaluator as feval


class cross_validation:
    def __init__(self, model, gt_fname, folds, test=True, tune=False):
        self.performances = []
        self.dev_pers = []
        self.model = model
        # self.model_paras = model_paras
        self.folds = folds
        self.gt_fname = gt_fname
        self.test = test
        self.tune = tune
        self.evaluators = []
        self.dev_evas = []
        if self.test:
            for i in range(self.folds):
                fname = self.gt_fname.replace('.csv', 'fold' + str(i) + '.csv')
                self.evaluators.append(feval.fold_evaluator(fname))
        if self.tune:
            for i in range(self.folds):
                fname = self.gt_fname.replace('.csv', 'fold' + str(i) + '_comp.csv')
                self.dev_evas.append(feval.fold_evaluator(fname))

    def validate(self):
        if self.test:
            del self.performances[:]
            generated_ranking = self.model.ranking()
            for i in range(self.folds):
                self.performances.append(self.evaluators[i].evaluate(generated_ranking))
        if self.tune:
            del self.dev_pers[:]
            generated_ranking = self.model.ranking()
            for i in range(self.folds):
                self.dev_pers.append(self.dev_evas[i].evaluate(generated_ranking))

    def testing(self, model_paras_folds):
        assert len(model_paras_folds) == self.folds, '#folds not equals to #model_paras given'
        del self.performances[:]
        for i in range(self.folds):
            self.model.model_para = model_paras_folds[i]
            generated_ranking = self.model.ranking()
            self.performances.append(self.evaluators[i].evaluate(generated_ranking))