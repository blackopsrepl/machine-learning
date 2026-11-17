# Course datasets

Each directory holds one course dataset under the course's own file names,
formats, column layouts, dtypes and array shapes, so every example runs the
course workflow unchanged.

| Dataset | Files | Course use | Contents here |
|---|---|---|---|
| `linear` | `data/ex1data1.txt` (97×2) | C1 W2 restaurant profits | generated |
| `houses` | `data/houses.txt` (100×5) | C1 W2 housing labs | generated |
| `logistic` | `data/ex2data1.txt` (100×3), `data/ex2data2.txt` (118×3) | C1 W3 admissions, microchips | generated |
| `digits` | `data/X.npy` (5000×400, f8, Fortran order), `data/y.npy` (5000×1, u1) | C2 W1/W2 digit recognition | generated |
| `evaluation` | `data/data_w3_ex*.csv`, `data/c2w3_lab2_data*.csv` | C2 W3 evaluation and diagnosis labs | generated |
| `heart` | `heart.csv` (918 rows, 12 columns) | C2 W4 tree ensembles | **original** |
| `kmeans` | `data/ex7_X.npy` (300×2), `bird_small.png` (128×128 RGB) | C3 W1 K-means, image compression | generated |
| `anomaly` | `data/X_part*.npy`, `data/X_val_part*.npy`, `data/y_val_part*.npy` | C3 W1 anomaly detection | generated |
| `collaborative` | `data/small_movies_{Y,R,X,W,b}.csv`, `data/small_movie_list.csv` | C3 W2 collaborative filtering | generated |
| `content` | `data/content_*.csv`, `data/content_*_header.txt` | C3 W2 content-based filtering | generated, fewer rows |

## Why most files are generated

The course's data files are DeepLearning.AI course material, and the movie
data derives from MovieLens, whose terms forbid redistribution without
permission. Those files are therefore replaced by stand-ins produced by
[`generate.rb`](generate.rb) from fixed seeds:

- Every generated file keeps the original file name, format, columns, dtype
  and shape. The digit file stores 500 images per class in label order, each a
  20×20 image in column-major order. The movie matrices are 4778 movies × 443
  users with 39253 ratings in half-star steps. The notebook's personal-rating
  movies sit at the notebook's row indices.
- Values come from documented generating models (noisy linear and quadratic
  relations, Gaussian clusters and anomalies, stroke-rendered digits, latent
  factor ratings). They exercise the same workflows; they are not the course's
  measurements, so learned parameters and accuracies differ from the notebooks.
- The content-based tables keep the course's 17-column user and item schema
  and one-row-per-(rating, genre) construction, with 30 users and 64 movies
  instead of 395 and 694. Training the original 58187-row table takes hours in
  pure Ruby.

Regenerate with `ruby data/generate.rb`. The output is deterministic.

## heart.csv

`heart/heart.csv` is the
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
by fedesoriano, distributed under the
[Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
It is included unmodified.

## Running on the original course files

With a copy of the course repository:

```sh
ruby data/import_course.rb /path/to/MachineLearningSpecialist /path/to/course-data
ML_DATA=/path/to/course-data make check
```

[`sources.json`](sources.json) lists each dataset's source directory and
files. `ML_DATA` points the examples at any tree with this layout; the code
does not change.
