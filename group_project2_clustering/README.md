# Group Assignment 2: Clustering

This project is a part of the course GNG5125. The goal of this assignment is to build a cluster model using various machine learning algorithms to predict the class of a given set of data.

## Requirements
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn

## Data
Choice five books have the same genres and semantically the same, download: https://www.gutenberg.org

    - "Peter Pan" by J. M. Barrie https://www.gutenberg.org/files/16/16-0.txt
    - "On_the_Origin_of_Species" by Charles Darwin https://www.gutenberg.org/cache/epub/1228/pg1228.txt
    - "War_and_Peace" by Leo Tolstoy https://www.gutenberg.org/cache/epub/2600/pg2600.txt
    - "Pride_and_Prejudice" by Jane Austen https://www.gutenberg.org/cache/epub/1342/pg1342.txt
    - "The_Adventures_of_Sherlock_Holmes" by Arthur Conan Doyle https://www.gutenberg.org/files/1661/1661-0.txt
    
## Implementation
The project is implemented in a Jupyter Notebook. The notebook contains the following sections:

1. Data Exploration
    - Load the data
    - Overview of the data
2. Preprocessing
    - Cleaning and preparation of the data
    - Operated by `data.ipynb`
3. Model Building
    - Split the data into training and testing sets
    - Build clustering models using various algorithms (Kmean,EM,Hierarchical,Word Embedding)
    - Train and evaluate the models
4. Model Evaluation
    - Comparison of the models based on Kappa,Silhouette,Coherence,Rand Index

## Usage
1. Clone the repository: `git clone https://github.com/zhouzhouce/GNG5125.git`
2. Navigate to the project folder: `cd GNG5125/group_project2_clustering`
3. Open the Jupyter Notebook: `data.ipynb` and `jupyter notebook main.ipynb`
4. Run the code cells in the notebook to see the results.

## Contributors
- Yunzhou Wang  300151213 
- Haiwei Nan    300250954
- Manjie Hou    300254157 
- Yuting Cao    8795048

## License
This project is licensed under the MIT License.
