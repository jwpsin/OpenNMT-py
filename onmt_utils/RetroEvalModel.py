from onmt_utils.EvalModel import *
from onmt_utils.evaluator import ONMTMultiEvaluator
from onmt_utils.metrics import *
import os


class RetroEvalModel(EvalModel):

    def __init__(self, datapath, dataset_name, split='test', n_best=1, class_sep=' '):
        EvalModel.__init__(self, datapath, dataset_name, split=split, n_best=n_best, class_sep=class_sep)
        self.__create_evaluators()

    def __create_evaluators(self):
        self.split_evaluator = ONMTMultiEvaluator(targets=self.precursors, sources=self.product)
        self.split_evaluator.df['tok_source'] = self.product_tok
        self.split_evaluator.df['tok_target'] = self.precursors_tok
        self.split_evaluator.df['class_ground_truth'] = self.classes
        self.split_evaluator.df['tok_class_ground_truth'] = self.classes_tok
        self.split_evaluator.df['superclass_ground_truth'] = [x[0] for x in self.classes]

    def add_experiment(self, results_path, fwd_results_path, class_results_path, fwd_log_likelihood=True,
                       func=get_names):
        names = EvalModel.add_experiment(self, results_path, func=func)

        # Enrich the dataframe with informations from the forward model
        for exp_name in names:
            # Add the class predictions
            classes = [[] for i in range(self.n_best)]
            classes_prediction_file = class_results_path + '/classes_rxn_' + self.name_basename_dict[
                exp_name] + '.out.txt'

            with open(classes_prediction_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    classes[i % self.n_best].append(line.strip().split(' '))

            # Add the forward predictions on the retro predictions
            predictions = [[] for i in range(self.n_best)]
            fwd_canonical_prediction_file = fwd_results_path + '/can_fwd_can_' + self.name_basename_dict[
                exp_name] + '.out.txt'

            with open(fwd_canonical_prediction_file, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    predictions[i % self.n_best].append(line.strip().replace(" ", ""))

            for n in range(1, self.n_best + 1):
                self.split_evaluator.df[f'fwd_{exp_name}_top_{n}'] = predictions[n - 1]

                self.split_evaluator.df[f'fwd_{exp_name}_top_{n}_valid'] = \
                    self.split_evaluator.df[f'fwd_{exp_name}_top_{n}'] != \
                    self.split_evaluator.invalid_smiles_replacement

                self.split_evaluator.df[f'fwd_{exp_name}_top_{n}_correct'] = \
                    self.split_evaluator.df[f'fwd_{exp_name}_top_{n}'] == \
                    self.split_evaluator.df['source']

                self.split_evaluator.df[f'classes_{exp_name}_top_{n}'] = classes[n - 1]

            if fwd_log_likelihood:

                log_probs_file = os.path.join(fwd_results_path,
                                              'fwd_can_' + self.name_basename_dict[exp_name] + '.out.txt_log_probs')
                self.split_evaluator.logger.info(f"Loading fwd log probabilities from {log_probs_file}")

                nll_predictions = [[] for i in range(self.n_best)]
                probas_predictions = [[] for i in range(self.n_best)]

                with open(log_probs_file, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        nll_predictions[i % self.n_best].append(-float(line.strip()))
                        probas_predictions[i % self.n_best].append(np.exp(float(line.strip())))

                for n in range(1, self.n_best + 1):
                    # Here topn refers to the retro topn prediction
                    self.split_evaluator.df[f'fwd_{exp_name}_top_{n}_nll'] = nll_predictions[n - 1]
                    self.split_evaluator.df[f'fwd_{exp_name}_top_{n}_prob'] = probas_predictions[n - 1]

    def __CJSD(self, exp_name, superclasses=range(12), x_min=0.5, nbins=100):

        cdfs = []
        x_grid = np.linspace(x_min, 1.0, nbins)

        for i in superclasses:
            data = []
            df = self.split_evaluator.df

            for n in range(1, self.n_best + 1):
                x = df[(df[f'fwd_{exp_name}_top_{n}_correct']) &
                       (df[f'classes_{exp_name}_top_{n}'].apply(lambda s: s[0]) == str(i)) &
                       (df[f'fwd_{exp_name}_top_{n}_prob'] >= x_min)][f'fwd_{exp_name}_top_{n}_prob'].values
                data.extend(x)

            if data:
                data_sorted = sorted(data)
                cdf = [1 / len(data) * i for i in range(1, len(data) + 1)]
                resampled_cdf = assign_left(data_sorted, cdf, x_grid)
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
        return self.__CJSD(exp_name, superclasses, x_min, nbins)

    def get_CJSD_allexp(self, superclasses=range(12), x_min=0.5, nbins=100):
        CJSDs = []
        sum_of_entropies = []
        H_devs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            cjsd, sum_of_entr, hdev = self.get_CJSD(exp_name,
                                                    superclasses=superclasses, x_min=x_min, nbins=nbins)
            CJSDs.append(cjsd)
            sum_of_entropies.append(sum_of_entr)
            H_devs.append(hdev)
        return CJSDs, sum_of_entropies, H_devs, labels

    def __class_diversity(self, exp_name):

        count_unique = []
        df = self.split_evaluator.df

        for i, row in df.iterrows():
            cols = [f'classes_{exp_name}_top_{i}' for i in range(1, self.n_best + 1) if
                    row[f'fwd_{exp_name}_top_{i}_correct']]
            count_unique.append(row[cols].apply(lambda x: x[0]).nunique())

        return np.mean(count_unique)

    def get_class_diversity(self, exp_name):
        return self.__class_diversity(exp_name)

    def get_class_diversity_allexp(self):
        CDs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            cd = self.get_class_diversity(exp_name)
            CDs.append(cd)
        return CDs, labels

    def __round_trip_accuracy(self, exp_name):

        cols = [f'fwd_{exp_name}_top_{i}_correct' for i in range(1, self.n_best + 1)]
        df_mean = self.split_evaluator.df[cols].mean(axis=1)

        return np.mean(df_mean) * 100

    def __round_trip_accuracy_v2(self, exp_name):

        cols = [f'fwd_{exp_name}_top_{i}_correct' for i in range(1, self.n_best + 1)]
        df_sum = self.split_evaluator.df[cols].sum(axis=1)

        return np.sum(df_sum) / len(self.split_evaluator.df) / self.n_best * 100

    def get_round_trip_accuracy(self, exp_name):
        return self.__round_trip_accuracy(exp_name)

    def get_round_trip_accuracy_v2(self, exp_name):
        return self.__round_trip_accuracy_v2(exp_name)

    def get_round_trip_accuracy_allexp(self):
        RTAs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            rta = self.get_round_trip_accuracy(exp_name)
            RTAs.append(rta)
        return RTAs, labels

    def get_round_trip_accuracy_allexp_v2(self):
        RTAs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            rta = self.get_round_trip_accuracy_v2(exp_name)
            RTAs.append(rta)
        return RTAs, labels

    def __coverage(self, exp_name):

        correct = 0
        df_filter = [True for i in range(len(self.split_evaluator.df))]

        for n in range(1, self.n_best + 1):
            correct += self.split_evaluator.df[df_filter][f'fwd_{exp_name}_top_{n}_correct'].sum()
            df_filter = df_filter & (self.split_evaluator.df[f'fwd_{exp_name}_top_{n}_correct'] == False)

        return correct / self.split_evaluator.total * 100

    def get_coverage(self, exp_name):
        return self.__coverage(exp_name)

    def get_coverage_allexp(self):
        COVs = []
        labels = []
        for exp_name in self.name_basename_dict:
            labels.append(exp_name)
            cov = self.get_coverage(exp_name)
            COVs.append(cov)
        return COVs, labels
