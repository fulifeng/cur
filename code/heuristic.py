from util_io import  read_ranking_list
class heuristic:
    def __init__(self, rl_fname):
        self.rl_fname = rl_fname

    def ranking(self):
        return read_ranking_list(self.rl_fname, dtype=float)