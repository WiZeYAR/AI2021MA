

class import_methods:
    """
    the class is responsable to import libraries and methods in the right directory,
    since that there is a conflict between colaboratory and local runs
    """
    def __init__(self, library):
        import os
        curr_dir = os.getcwd()

        if 'AI' in curr_dir:
            if 'solvers' in library:
                from solvers.utils import *
                from solvers.constructive_algorithms import *
                from solvers.local_search import *

            if 'IO' in library:
                pass

            if 'run' in library:
                import
        else:
            if 'solvers' in library:
                from AI2020.solvers.utils import *
                from AI2020.solvers.constructive_algorithms import *
                from AI2020.solvers.local_search import *

