from configs.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)
        parser.add_argument('--result_dir', type=str, default='./exp/results',
                            help='save result images into result_dir/exp_name')
        parser.add_argument('--ckpt_dir',   type=str,
                            default='./ckpt/best_model_season.ckpt', 
                            help='load ckpt path')    
        parser.add_argument('--save_eval_pngs', action='store_true',
                            help='save result image into evaluation form')
        
        return parser


