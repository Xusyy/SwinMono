import argparse


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='/mnt/D/xusy/dataset/SeasonDepth')
        parser.add_argument('--dataset',      type=str, default='seasondepth')
        parser.add_argument('--exp_name',     type=str, default='test')
        parser.add_argument('--batch_size',   type=int, default=1)
        parser.add_argument('--workers',      type=int, default=2)
        
        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
              
        return parser
