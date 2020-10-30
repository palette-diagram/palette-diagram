import os
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph_shortest_path import *

from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'IPAGothic'

def export_palette_diagram(palette_type):
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    plt.savefig('./output/'+palette_type+'_palette_diagram.pdf')


def export_ordering(df, order, angles):
    data_index = df.index.tolist()

    columns = ['data_index','ordering'] if len(angles)==0 else ['data_index','ordering','angle']
    table = np.zeros((order.size, len(columns)))
    for i, odr in enumerate(order):
        table[i, 0] = data_index[i]
        table[i, 1] = odr

    df_table = pd.DataFrame(table, columns=columns)
    df_table = df_table.sort_values('ordering')
    df_table = df_table.drop('ordering', axis=1)

    if len(angles) > 0:
        angle_degree = [180*angle/np.pi for angle in angles]
        df_table['angle'] = angle_degree

    df_table = df_table.set_index('data_index')
    df_table.to_csv('./output/data_ordering.csv')

#---------------------------------------
#-------- color palettes ---------------
#---------------------------------------
def qualitative_color_pallete_5(M):
    five_colors = ['#008744','#ffa700','#d62d20','#0057e7','#494949']

    color_palette = []
    for k in range(M):
        color_palette.append(five_colors[k])

    return color_palette

def qualitative_color_pallete_cmap(M, cmap_name):
    cm = plt.get_cmap(cmap_name)

    color_palette = []
    for k in range(M):
        color_palette.append(cm(k))

    return color_palette

def equidistant_color_pallete(M):
    cmap_1d = np.linspace(0, 1, M+1)[:-1]
    LL = .6
    SS = .8

    color_palette = []
    for sim in cmap_1d:
        color_palette.append( sns.hls_palette(1, h=sim, l=LL, s=SS)[0] )

    return color_palette


#---------------------------------------
#---------------------------------------


#---------------------------------------
#-------- dimension reduction ----------
#---------------------------------------
def isomap(X,n_neighbors,n_components=1):
	Y = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X)
	order = np.argsort(Y.reshape(1,-1)[0])
	return order, Y

def manifold_learning(data, n_neighbors):
	n = data.shape[0]
	if n < n_neighbors:
		n_neighbors = int(n/2)

	order, Y = isomap(data, n_neighbors)

	return data[order,:], order

@jit('f8[:](f8[:,:],i4,i4,f8)' , nopython=True)
def sammons_nonlinear_mapping(D, N, n_epochs, lr):
    z = np.random.randn(N)
    for epoch in range(n_epochs):

        i_list = np.random.randint(0, N, (N,))
        j_list = np.random.randint(0, N, (N,))

        for i, j in zip(i_list, j_list):
            if i == j: continue

            grad1 = 0
            grad2 = 0
            for k in range(N):
                grad1 += (  D[i,k] - (1 - np.cos(z[i] - z[k]))  ) * np.sin(z[i] - z[k])
                grad2 += (  D[k,j] - (1 - np.cos(z[k] - z[j]))  ) * np.sin(z[k] - z[j])

            z[i] += lr * grad1
            z[j] -= lr * grad2

    return z % (2*np.pi)

def periodic_manifold_learning(X, n_neighbors, n_epochs, lr):

    N = X.shape[0]
    X = X / np.sqrt(np.sum(X ** 2, axis=1)).reshape(-1, 1)
    X = np.nan_to_num(X)

    A = kneighbors_graph(X, n_neighbors, mode='distance')
    D = graph_shortest_path(A, method="D", directed=False)

    z = sammons_nonlinear_mapping(D, N, n_epochs, lr)

    return z
#---------------------------------------
#---------------------------------------



#---------------------------------------
#-------- draw palette diagrams --------
#---------------------------------------
def remove_all_zeros(df, axis):
    df_s = df.sum(axis=(axis+1)%2)
    index = df_s.index[df_s.values == 0]
    df = df.drop(index, axis=axis)
    return df

def preprocessing(df, remove_empty_sets):
    df = df.fillna(0)
    if (remove_empty_sets == 0) or (remove_empty_sets == 2):
        df = remove_all_zeros(df, axis=0)
    if (remove_empty_sets == 1) or (remove_empty_sets == 2):
        df = remove_all_zeros(df, axis=1)
    return df

def circular_palette_diagram(df, n_neighbors, n_epochs, lr, norm, export, export_table, category_labels, cmap_name):

    if norm == True:
        df = df.div(df.sum(axis=1), axis=0)

    X = df.values

    Z = periodic_manifold_learning(X, n_neighbors, n_epochs, lr)
    order = np.argsort(Z.reshape(1, -1)[0])
    X = X[order, :]
    # ////////////////////////

    angles = ((2 * np.pi) / (X.shape[0])) * np.arange(X.shape[0])

    areas = np.sum(X, axis=0)
    layer_order = np.argsort(areas)[::-1]
    X = X[:, layer_order]

    category_labels_internal = df.columns.tolist()
    category_labels_internal = np.array(category_labels_internal)[layer_order]
    if category_labels == None:
        category_labels = df.columns.tolist()

    if cmap_name != None:
        color_palette = qualitative_color_pallete_cmap(len(category_labels), cmap_name)
    else:
        if len(category_labels) <= 5:
            color_palette = qualitative_color_pallete_5(len(category_labels))
        else:
            color_palette = equidistant_color_pallete(len(category_labels))

    color_codes = {name:color_palette[idx] for idx,name in enumerate(category_labels)}
    color_list = np.array([color_codes[name] for name in category_labels_internal])

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, polar=True)
    ax.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
    ax.xaxis.grid(False)
    width_blank = 0.3
    widh_first_layer_factor = 0.8
    if norm == False:
        yticks_width = np.max(X, axis=0)
        width_first_layer = np.sum(yticks_width) / X.shape[1] * widh_first_layer_factor
        width_blank = width_blank
        yticks_width = np.insert(yticks_width, 0, width_blank)
        yticks_width = np.insert(yticks_width, 0, width_first_layer)
        yticks_width = np.insert(yticks_width, 0, 0)
        yticks = np.cumsum(yticks_width)
    else:
        width_first_layer = 1 * widh_first_layer_factor
        yticks_width = np.ones(X.shape[1])
        yticks_width = np.insert(yticks_width, 0, width_blank)
        yticks_width = np.insert(yticks_width, 0, width_first_layer)
        yticks_width = np.insert(yticks_width, 0, 0)
        yticks = np.cumsum(yticks_width)
    ax.set_yticks(yticks)
    ax.set_thetagrids(range(0, 360, 30))
    ax.spines['polar'].set_visible(False)

    central_color_palette = color_list[np.argmax(X, axis=1)]
    for i in range(X.shape[1] + 1):
        if i == 0:
            for n in range(X.shape[0]):
                y1 = np.zeros(2)
                y2 = np.ones(2) * width_first_layer
                plt.fill_between([angles[n],angles[(n+1)%X.shape[0]]], y1, y2, facecolor=central_color_palette[n], alpha=0.7)
            continue

        y1 = np.ones(angles.size) * yticks[i+1]
        y2 = y1 + X[:, i - 1]
        plt.fill_between(angles, y1, y2, facecolor=color_list[i - 1], alpha=0.7, label=category_labels_internal[i-1])
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=16)

    if export == True:
        plt.tight_layout()
        export_palette_diagram(palette_type='circular')
    if export_table == True:
        export_ordering(df, order, angles=angles)

    # plt.show()


def linear_palette_diagram(df, n_neighbors, norm, export, export_table, category_labels, cmap_name):
    category_labels_internal = df.columns.tolist()
    if category_labels == None:
        category_labels = df.columns.tolist()

    if cmap_name != None:
        color_palette = qualitative_color_pallete_cmap(len(category_labels), cmap_name)
    else:
        if len(category_labels) <= 5:
            color_palette = qualitative_color_pallete_5(len(category_labels))
        else:
            color_palette = equidistant_color_pallete(len(category_labels))

    color_codes = {name:color_palette[idx] for idx,name in enumerate(category_labels)}
    color_list = [color_codes[name] for name in category_labels_internal]
    
    if norm == True:
        df = df.div(df.sum(axis=1), axis=0)

    X, order = manifold_learning(df.values, n_neighbors)

    fig, ax = plt.subplots(1,1,sharex=True, figsize=(12,4))
    ax.stackplot(range(X.shape[0]), X.T, labels=category_labels_internal, 
                  baseline='wiggle', colors=color_list, alpha=0.7)
    ax.get_yaxis().set_visible(False)
    plt.xticks(rotation=0, fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=12)
    plt.tight_layout()

    if export == True:
        plt.tight_layout()
        export_palette_diagram(palette_type='linear')
    if export_table == True:
        export_ordering(df, order, angles=[])

    # plt.show()

#---------------------------------------
#---------------------------------------


#-------- main --------
def palette_diagram(df, palette_type='circular', n_neighbors=100, n_epochs=100, lr=0.0005, norm=True, export=True, export_table=True, category_labels=None, cmap_name=None, remove_empty_sets=-1):
    df = preprocessing(df, remove_empty_sets)

    n_neighbors_ = n_neighbors if n_neighbors < len(df) else len(df)-1
    if n_neighbors >= len(df):
        print('Warning: n_neighbors is larger than or equal to the number of data elements!')
        print('n_neighbors was replaced with len(df)-1.')

    if palette_type == 'circular':
        circular_palette_diagram(df, n_neighbors_, n_epochs, lr, norm, export, export_table, category_labels, cmap_name)

    elif palette_type == 'linear':
        linear_palette_diagram(df, n_neighbors_, norm, export, export_table, category_labels, cmap_name)

    else:
        print('No option called '+palette_type+' exists.')

    