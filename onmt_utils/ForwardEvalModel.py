from onmt_utils.EvalModel import *
from onmt_utils.evaluator import ONMTMultiEvaluator
from onmt_utils.metrics import *


class ForwardEvalModel(EvalModel):
    def __init__(self, datapath, dataset_name, split='test', n_best=1, class_sep=" "):
        EvalModel.__init__(self, datapath, dataset_name, split=split, n_best=n_best, class_sep=class_sep)
        self.__create_evaluators()

    def __create_evaluators(self):

        self.split_evaluator = ONMTMultiEvaluator(targets=self.product, sources=self.precursors)
        self.split_evaluator.df['tok_source'] = self.precursors_tok
        self.split_evaluator.df['tok_target'] = self.product_tok
        self.split_evaluator.df['class_ground_truth'] = self.classes
        self.split_evaluator.df['tok_class_ground_truth'] = self.classes_tok
        self.split_evaluator.df['superclass_ground_truth'] = [x[0] for x in self.classes]

    def add_experiment(self, results_path, classification=False, class_results_path='', func=get_names):

        names = EvalModel.add_experiment(self, results_path, func=func)

        # Add the PREDICTED classes, optional
        if classification:
            for exp_name in names:
                classes = [[] for i in range(self.n_best)]
                classes_prediction_file = class_results_path + '/classes_rxn_' + self.name_basename_dict[
                    exp_name] + '.out.txt'

                with open(classes_prediction_file, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        classes[i % self.n_best].append(line.strip().split(' '))

                for n in range(1, self.n_best + 1):
                    self.split_evaluator.df[f'class_{exp_name}_top_{n}'] = classes[n - 1]
                    self.split_evaluator.df[f'class_{exp_name}_top_{n}_correct'] = self.split_evaluator.df[
                                                                                       f'class_{exp_name}_top_{n}'] == \
                                                                                   self.split_evaluator.df[
                                                                                       'class_ground_truth']

    def __CJSD(self, exp_name, superclasses=range(12), x_min=0.5, nbins=100):

        # This definition of JSD uses the CUMULATIVE DISTRIBUTIONS and applies the
        # integral version of entropy.

        cdfs = []
        samples = []
        x_grid = np.linspace(x_min, 1.0, nbins)

        for i in superclasses:
            data = []
            df = self.split_evaluator.df

            for n in range(1, self.n_best + 1):
                x = df[(df[f'{exp_name}_top_{n}_correct']) &
                       (df[f'superclass_ground_truth'] == str(i)) &
                       (df[f'{exp_name}_top{n}_prob'] >= x_min)][f'{exp_name}_top{n}_prob'].values
                data.extend(x)

            if data:
                data_sorted = sorted(data)
                cdf = [1 / len(data) * i for i in range(1, len(data) + 1)]
                resampled_cdf = assign_left(data_sorted, cdf, x_grid)
                samples.append(data_sorted)
            else:
                print(f'Data not found for class {i}')
                resampled_cdf = np.zeros(len(x_grid))

            cdfs.append(resampled_cdf)

        weights = np.empty(len(superclasses))
        weights.fill(1 / len(superclasses))

        JSD, entropy_of_mixture, sum_of_entropies = cumulative_jensen_shannon_divergence(cdfs, weights, x_grid)
        H_dev = [sum_of_entropies - CRE(cdfs[i], x_grid) for i in range(len(cdfs))]
        return JSD, sum_of_entropies, H_dev

    def get_CJSD(self, exp_name, superclasses=range(12), x_min=0.5, nbins=100):
        return self.__CJSD(exp_name=exp_name, superclasses=superclasses, x_min=x_min, nbins=nbins)

    def get_CJSD_allexp(self, superclasses=range(12), x_min=0.5, nbins=100):
        CJSDs = []
        sum_of_entropies = []
        H_devs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            cjsd, sum_of_entr, hdev = self.get_CJSD(exp_name, superclasses=superclasses, x_min=x_min, nbins=nbins)
            CJSDs.append(cjsd)
            sum_of_entropies.append(sum_of_entr)
            H_devs.append(hdev)
        return CJSDs, sum_of_entropies, H_devs, labels
