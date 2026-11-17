# machine-learning

Andrew Ng's Python Machine Learning Specialization, reimplemented as small,
readable Ruby demonstrations. The format matches [heuristics](https://github.com/blackopsrepl/heuristics):
one inspectable file per algorithm, shared toy problems, explicit arithmetic,
deterministic traces, pure Ruby and zero gems or native extensions.

The course defines the scope, not the implementation language. Framework-trained
algorithms are reimplemented here; this is not a wrapper around TensorFlow,
scikit-learn or XGBoost. Examples use deliberately small educational fixtures,
not copied Coursera datasets or trained models.

```sh
make              # menu
make all          # every implemented example
make supervised   # one family
make check        # syntax and executable numerical checks
```

## Layout

`lib/problem.rb` holds shared toy data, `lib/numeric.rb` elementary array
arithmetic and small linear solves, and `lib/style.rb` the ANSI vocabulary
shared with heuristics. Learning algorithms live in their named example files.
Each file can be required without running its demonstration.
