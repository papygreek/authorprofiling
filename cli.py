import argparse
from authorprofiling import authorprofiling

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_known_args(f.read().split(), namespace)

parser = argparse.ArgumentParser(
    prog='PapyGreek Author Profiling',
    description='Tools for profiling authors and writers in PapyGreek data',
    epilog='Thank you'
)

parser.add_argument('-target', choices=['writer', 'author'], default='author')
parser.add_argument('-stats', nargs='?', const=10, type=int, default=0)
parser.add_argument('-persons', nargs='*', type=str, default=False)
parser.add_argument('-text_name_contains', type=str, default=None)
parser.add_argument('-text_name_not_contains', type=str, default=None)
parser.add_argument('-min_vars',  type=int, default=0)
parser.add_argument('-min_tokens', type=int, default=0)
parser.add_argument('-features', nargs='*', default=[])
parser.add_argument('-model', choices=['word2vec', 'fasttext'], default='fasttext')
parser.add_argument('-plot', choices=['2d', '3d'], default='2d')
parser.add_argument('-treebanks', action='store_true')
parser.add_argument('-arrows', action='store_true')
parser.add_argument('-clusters', type=int, default=0)
parser.add_argument('-mode', choices=['clustering', 'classification'], default="clustering")
parser.add_argument('--file', '--f', type=open, action=LoadFromFile, default=argparse.SUPPRESS)
authorprofiling.run(*parser.parse_known_args())
