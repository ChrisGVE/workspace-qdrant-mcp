//! Language-specific chunk extractors.

mod rust;
mod python;
mod javascript;
mod typescript;
mod go;
mod java;
mod c;
mod cpp;

pub use rust::RustExtractor;
pub use python::PythonExtractor;
pub use javascript::JavaScriptExtractor;
pub use typescript::TypeScriptExtractor;
pub use go::GoExtractor;
pub use java::JavaExtractor;
pub use c::CExtractor;
pub use cpp::CppExtractor;
