import argparse
from authorprofiling import authorprofiling

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

parser = argparse.ArgumentParser(
    prog='PapyGreek Author Profiling',
    description='Tools for profiling authors and writers in PapyGreek data',
    epilog='Thank you'
)

parser.add_argument('-target', '-t', choices=['writer', 'author'], help='The target: writer or author', default='author')
parser.add_argument('-stats', '-s', nargs='?', const=10, type=int, default=0)
parser.add_argument('-persons', '-p', nargs='*', type=str, default=False)
parser.add_argument('-min_vars', '-mv', type=int, default=0)
parser.add_argument('-min_tokens', '-mt', type=int, default=0)
parser.add_argument('-features', '-f', nargs='*', default=[])
parser.add_argument('-algorithm', '-a', choices=['LSA', 'PCA', 'TSNE'], default='PCA')
parser.add_argument('-model', '-m', choices=['word2vec', 'fasttext'], default='fasttext')
parser.add_argument('-treebanks', '-tb', action='store_true')
parser.add_argument('-clusters', '-c', type=int, default=1)
parser.add_argument('--file', '--f', type=open, action=LoadFromFile, default=argparse.SUPPRESS)
authorprofiling.run(parser.parse_args())
