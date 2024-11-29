import sys
import numpy as np
import matplotlib.pyplot as plt

# define IndexTracker class to map events and scores
class ScoringPanel(object):
    def __init__(self, fig, ax, X, Y, event_label, fn_ch, start_ind):
        self.fig = fig
        self.ax = ax
        self.X = X
        self.Y = Y
        self.event_label = event_label # adding the type of event label
        self.fn_ch = fn_ch # adding the filename and channel
        self.ind = start_ind
        self.col_dict = {0:'r', 1:'k', 2:'b'}
        self.update()

    # Use mouse to change event
    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1)
        else:
            self.ind = (self.ind - 1)
        self.update()

    # Use keyboard to change events and scores
    def onclick(self, event):
        # print('press', event.key) ### Can be used to show key bindings
        sys.stdout.flush()
        if event.key == 'right':
            self.ind = (self.ind + 1)
        elif event.key == 'left':
            self.ind = (self.ind - 1)
        if event.key == 'm':
            self.Y[self.ind] = (self.Y[self.ind]+1)%3
        self.update()


    def update(self):
        self.ax.clear()
        
        # Take care of boundary indices
        if self.ind < 0:
            self.ind += self.Y.shape[0]
        elif self.ind >= self.Y.shape[0]:
            self.ind = self.ind - self.Y.shape[0]
           
        self.ax.plot(self.X[self.ind], c=self.col_dict[self.Y[self.ind]])
        self.ax.set_ylabel('# %s' % self.ind)
        # self.ax.set_title(f'label: {self.Y[self.ind]}')
        self.ax.set_title(f'label: {self.Y[self.ind]}, event type: {self.event_label[self.ind]}, \
                          fn_ch: {self.fn_ch[self.ind]}') #  cell-label, quality label

        self.fig.canvas.draw_idle()