# Software Analysis Examples

Contains examples from the course on Software Analysis at the
[University of Passau](https://www.uni-passau.de).  The examples are mainly toy 
examples to visualise a certain aspect during the course.  They are *not* meant to 
be complete or perfect; they also do not necessarily meet a certain level of style 
or code quality!

Feel free to fork this repository and file a pull request to add further examples.

## The Examples

* `ast_example` provides an AST visitor and an AST rewriter based on Python's AST 
  module.  It is using the [`astor`](https://astor.readthedocs.io) library to 
  extract source code from the AST again.
* `controldependencies` is an implementation of control-flow graph, dominator and 
  post-dominator tree, and control-dependence graph based on Python bytecode.  It 
  emits Graphviz DOT files of the graphs and automatically converts them to PNG.  
  Make sure you have the `dot` utility installed and available on your `PATH`.
* `mocking` together with its test module shows some aspects of how to use mocking 
  in Python.
