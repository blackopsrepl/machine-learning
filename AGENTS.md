# Machine Learning Laboratory

## Scope

The Python Andrew Ng Machine Learning Specialization defines the algorithm list.
Use `docs/course-map.md` to check assignments and optional labs. Do not import
Octave-only algorithms or turn the repository into a general ML framework.

## Format

Match the sibling `../heuristics` laboratory's presentation: one inspectable Ruby
file per algorithm, a concise theory header and step-by-step terminal traces.
Preserve the course's problems, dataset shapes, architectures, preprocessing and
training settings; only plots and widgets become terminal output. Share the ANSI
vocabulary and expose `make`, family targets and individual algorithm targets.

Keep learning mathematics in named algorithm files, not hidden behind `lib/`.
Guard demonstrations with `if __FILE__ == $PROGRAM_NAME` so requiring an
implementation has no output or training side effects. Seed randomized
demonstrations explicitly.

## Self-contained Ruby

The repository runs on Ruby alone: no Python, no gems, no downloads, no external
course checkout. Where the course calls a library (scikit-learn, Keras, XGBoost,
Gym/Box2D), port that library's algorithm and defaults to Ruby, including its
random streams, rather than writing a narrower substitute. Course data ships in
`data/` under the course's names and shapes; see `data/README.md`. Never hard-code
dataset dimensions: the original files must work when dropped in via `ML_DATA`.

## Verification and commits

`make check` runs the stream and dataset checks, a syntax pass and every example
at full course budgets. Keep numerical checks with the behavior they verify;
gradient checks compare the actual training engine against finite differences.
Validate coverage against the course map and the README table, not file counts.

Work in coherent atomic conventional commits with meaningful bodies. Each commit
must execute from a clean committed snapshot.
