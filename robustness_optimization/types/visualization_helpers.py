import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_design(state):
    '''
    returns pairplots of design
    '''
    plt.figure()
    fig = sns.pairplot(pd.DataFrame(state))
    return fig