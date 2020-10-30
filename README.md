# Palette diagram
A palette diagram is a visualization tool for a large number of categorical datasets, each comprising several categories.

![schmatic](https://raw.githubusercontent.com/palette-diagram/palette-diagram/main/img/Illustration.png)

### Linear palette diagram 

![linear](https://raw.githubusercontent.com/palette-diagram/palette-diagram/main/img/linear_palette_diagram.png) 

This is a stream plot, which is usually used for plotting time series data. 
Each categorical dataset is stacked vertically, and these stacked plots are aligned horizontally so that the neighboring datasets have similar vertical patterns. 


### Circular palette diagram 

![circula](https://raw.githubusercontent.com/palette-diagram/palette-diagram/main/img/circular_palette_diagram.png) 

Each categorical dataset is represented along the radial coordinate: Each layer corresponds to a category and the thickness represents the normalized (or unnormalized) quantity within a category.
The set of categorical data is aligned along the angular coordinate. 

The central part shows indicates the color of the dominant category within a categorical dataset (i.e., the maximum a posteriori estimate). 


## Installation
`palette-diagram` can be installed from PyPI:

```
pip install palette-diagram
```

## Usage
This function generates a linear or circular palette diagram from a data table. 

```python
palette_diagram(df,
	palette_type='circular',
	n_neighbors=100,
	n_epochs=100,
	lr=0.0005,
	norm=True,
	export=True,
	export_table=True,
	category_labels = None,
	cmap_name = None,
	remove_empty_sets=-1)
```



### input

A data table `df` in *pandas* DataFrame. 
Each row represents a categorical data of a data element. 
Here is how the DataFrame should look like: 

|  | category A | category B | category C | category C |
| :--- | :---: | ---: | ---: | ---: |
| 0 | 15 | 31 | 2 | 8 |
| 1 | 24 | nan | 45 | 112 |
| 2 | 9 | 11 | 83 | 0 |
| ... | ... | ... | ... | ... |

- The (i,k) element in the DataFrame represents a quantity for *k*th category in *i*th dataset. 
- The DataFrame must have column names representing the category labels. 
- The indices (the first column) will be used as the dataset ID in `data_ordering.csv`
- The value of each cell has to be non-negative.
- A dataset is allowed to have missing cells (`nan`). The missing cells are filled with zeros.

### output

A linear palette diagram or circular palette diagram.

### Optional parameters

|argument|type|description|
|:--|--:|:--:|
| palette_type |'circular', 'linear'|  'circular': circular palette diagram </br> 'linear': linear palette diagram|
| n_neighbors | integer | A hyperparameter for the linear palette diagram (see below for details) |
| n_epochs | integer | A hyperparameter for the circular palette diagram (see below for details) |
| lr | float | A hyperparameter for the circular palette diagram (see below for details) |
| norm | boolean | If True, each categorical dataset (row in the datamatrix) will be normalized to unity. The diagram has non-uniform layer thickness when `norm=False`. |
| export | boolean | If True, the palette diagram will be saved as a PDF file in `./output/`. |
| export_table | boolean | If True, the ordering of the datasets will be saved as a csv in `./output/`. |
| category_labels | list of category labels | If provided, you can manually control the color assignment of each category by specifying the order of category labels. |
| cmap_name | string | If a qualitative color palette in *matplotlib* is provided, the specified color map is will be used. |
| remove\_empty\_sets | {0,1,2,-1} | 0: Remove all empty (zero-valued) rows (data) </br> 1: Remove all empty (zero-valued) columns (categories) </br> 2: Remove all empty (zero-valued) rows and columns </br> -1: Ignored |





## Order optimization
### linear palette diagram
In the linear palette diagram, the order of the datasets are optimized through ISOMAP. 
`n_neighbors` is a hyperparameter used to construct a k-nearest neighbor graph in ISOMAP. 

### circular palette diagram
In the circular palette diagram, the stochastic gradient descent (SGD) method is used for the order optimization.
`n_epochs` and `lr` are hyperparameters for the SGD: `n_epochs` is the number of epochs and `lr` is the learning rate.

We strongly recommend users to try various values of these hyperparameters, as the appropriate value varies depending on the input data table.





## References
- Please cite the following paper when you use the palette diagram: 

Chihiro Noguchi and Tatsuro Kawamoto, "Palette diagram: A Python package for visualization of collective categorical data," *in preparation*, (2020).

- You can find more details about the (linear) palette diagram in the following article: 

Chihiro Noguchi and Tatsuro Kawamoto, "Evaluating network partitions through visualization," arXiv:1906.00699, *unpublished* (2019).