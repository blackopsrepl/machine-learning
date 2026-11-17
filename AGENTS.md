# Machine Learning Laboratory

## Scope

The Python Andrew Ng Machine Learning Specialization defines the algorithm list.
Use `docs/course-map.md` to check assignments and optional labs. Do not import
Octave-only algorithms or turn the repository into a general ML framework.

## Format

Match the sibling `../heuristics` laboratory's presentation: one inspectable Ruby
file per algorithm, a concise theory header and step-by-step terminal traces.
Preserve the Python course's actual datasets, applications, architectures and
training workflows. Small fixtures may supplement them, never replace them. Share the ANSI
vocabulary and expose `make`, family targets and individual algorithm targets.

Keep learning mathematics in named algorithm files, not hidden behind `lib/`.
Composed models may require those algorithm files. Guard demonstrations with
`if __FILE__ == $PROGRAM_NAME` so requiring an implementation has no output or
training side effects. Seed randomized demonstrations explicitly.

Keep handwritten course algorithms in Ruby and identify them separately from the
library-backed course examples. The latter use the actual reference libraries via
PyCall, not narrower homemade substitutes. Use the original LunarLander-v2 simulator.
Install external runtimes only in the ignored .reference directory. Do not smuggle
Python helper scripts into this Ruby repository.

## Verification and commits

Run the real course-data demonstrations with `make check`. Exercise bounded training
through the real Lunar Lander entry point; never count a toy environment as coverage. Keep
numerical checks with the behavior they verify; gradient checks must compare the
actual training implementation against independent finite differences. Validate
coverage against the course map and the README matrix, not file counts alone.

Work in coherent atomic conventional commits with meaningful bodies. Each commit
must execute from a clean committed snapshot. Do not publish remotes, tags or
releases unless publication is authorized.
