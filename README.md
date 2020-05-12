# plydough
Ply file parser for Rust.

* Supports Ascii and Binary formats with any element structure

* Aims to be high-performance for loading large meshes by using
the [nom]() parser combinator library (including credit to the 
[lexical](https://docs.rs/lexical/5.2.0/lexical/) library for fast ascii decimal parsing).

* Avoids memory overhead of "dynamic" ply types by providing element data in
a struct-of-arrays format.

The next main goal for this crate is to provide a proc-macro based way of specifying
an "expected" struct to parse into directly, allowing simpler user code and potentially
better performance.
