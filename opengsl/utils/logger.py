import torch

class Logger(object):
    """
    Logger Class.

    Parameters
    ----------
    runs : int
        Total experimental runs.
    """
    def __init__(self, **kwargs):
        self.stats = {}
        self.agg_stats = {}

    def add_result(self, run, result_dict):
        '''
        Add performance of a new run.

        Parameters
        ----------
        run : int
            Id of the new run.
        result_dict : dict
            A dict containing training, valid and test performances.

        '''
        for key, value in result_dict.items():
            if key in self.stats.keys():
                self.stats[key].append(value)
            else:
                self.stats[key] = [value]

    def aggregate(self):
        for key, value in self.stats.items():
            if key in ['train', 'valid', 'test']:
                r = 100 * torch.tensor(self.stats[key])
            else:
                r = torch.tensor(self.stats[key])
            mean, std = r.mean(), r.std()
            self.agg_stats[key] = f'{mean:.2f} ± {std:.2f}'
            self.agg_stats[key+'_mean'] = mean.item()
            self.agg_stats[key + '_std'] = std.item()

    def print_statistics(self, run=None):
        '''
        Function to output the statistics.

        Parameters
        ----------
        run : int
            Id of a run. If not specified, output the statistics of all runs.

        Returns
        -------
            The statistics of a given run or all runs.

        '''
        self.aggregate()
        print(f'All runs:')
        print(f'Highest Train: ' + self.agg_stats['train'])
        print(f'Highest Valid: ' + self.agg_stats['valid'])
        print(f'   Final Test: ' + self.agg_stats['test'])
        
        try:
            import nni
            if nni.get_trial_id()!="STANDALONE":
                nni.report_final_result(float(self.agg_stats['test_mean']))
        except ImportError:
            pass
        
        return self.agg_stats['test_mean'], self.agg_stats['test_std']