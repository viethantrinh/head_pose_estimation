import yaml
from config import config_parser_p1
from utils.util import logger
from processor.processor_test_p1 import ProcessorTest

if __name__ == "__main__":
    parser = config_parser_p1.get_parser()  # get the parser for config
    # parse the data to Namespace object which use Dict to hold the key and value
    args = parser.parse_args()

    if args.config is not None:  # if the config value is not None
        with open(args.config, 'r') as f:  # get the yaml file which is the config file
            # and using yaml lib to load the config file into a Dict variable
            default_arg = yaml.safe_load(f)

        key = vars(args).keys()  # take all the key from original args

        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
            # parse the key and value of config file to the parser (override)
            parser.set_defaults(**default_arg)

        # get the final Namespace after override from config file
        final_arg = parser.parse_args()
        processor = ProcessorTest(final_arg)
        processor.start()
