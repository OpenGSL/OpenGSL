import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result_dict):
        assert "train" in result_dict.keys()
        assert "valid" in result_dict.keys()
        assert "test"  in result_dict.keys()
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result_dict["train"])
        self.results[run].append(result_dict["valid"])
        self.results[run].append(result_dict["test"])

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[0]:.2f}')
            print(f'Highest Valid: {result[1]:.2f}')
            print(f'   Final Test: {result[2]:.2f}')
        else:
            best_result = 100 * torch.tensor(self.results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')