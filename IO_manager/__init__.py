import os

if 'AI' in os.getcwd():
    from src.utils import *
    from src.constructive_algorithms import *
    from src.local_search import *
    from src.meta_heuristics import *
    from src.TSP_solver import *
    from src.io_tsp import *

else:
    from AI2019.src.utils import *
    from AI2019.src.constructive_algorithms import *
    from AI2019.src.local_search import *
    from AI2019.src.meta_heuristics import *
    from AI2019.src.TSP_solver import *
    from AI2019.src.io_tsp import *
