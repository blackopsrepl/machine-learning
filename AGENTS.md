# Machine Learning Laboratory

## Scope

The Python Andrew Ng Machine Learning Specialization defines the algorithm list.
Use `docs/course-map.md` to check assignments and optional labs. Do not import
Octave-only algorithms or turn the repository into a general ML framework.

## Format

Match the sibling `../heuristics` laboratory: pure Ruby, zero runtime dependencies,
one inspectable file per algorithm, a concise theory header, a tiny explanatory
fixture and a deterministic step-by-step terminal demonstration. Share the ANSI
vocabulary and expose `make`, family targets and individual algorithm targets.

Keep learning mathematics in named algorithm files, not hidden behind `lib/`.
Composed models may require those algorithm files. Guard demonstrations with
`if __FILE__ == $PROGRAM_NAME` so requiring an implementation has no output or
training side effects. Seed randomized demonstrations explicitly.

No TensorFlow, PyTorch, scikit-learn, XGBoost bindings, downloaded models or opaque
optimizer calls. Framework-based course models are reimplemented mathematically;
source/data substitutions must remain explicit in the README and course map.

## Verification and commits

Run the real demonstrations with `make check RUBY='ruby --disable-gems'`. Keep
numerical checks with the behavior they verify; gradient checks must compare the
actual training implementation against independent finite differences. Validate
coverage against the course map and the README matrix, not file counts alone.

Work in coherent atomic conventional commits with meaningful bodies. Each commit
must execute from a clean committed snapshot. Do not publish remotes, tags or
releases unless publication is authorized.
