import pickle

# # figx = pickle.load(open('/Volumes/alzheimer/verjinia/data/pickle/FigureObject.fig.pickle', 'rb'))

# # figx.show() # Show the figure, edit it, etc.!


# with open('/Volumes/alzheimer/verjinia/data/pickle/FigureObject.fig.pickle', 'rb') as file: 
#     figx = pickle.load(file)

# figx.show()

import svgutils.compose as sc
from IPython.display import SVG

sc.Figure("8cm", "8cm", 
    sc.Panel(sc.SVG("/Volumes/alzheimer/verjinia/miniML_multipatch/prediction.svg"))).save("compose.svg")
SVG('compose.svg')
