# Rusty Embeddings

**Rusty Embeddings** uses embedding tools to convert text into vector representations with persistent caching for fast reuse.

---

## Usage Guide

To include this crate in your project, add it to your dependencies:

```bash
cargo add --git https://github.com/sindre0830/rusty-embeddings.git --tag v1.0.0 rusty-embeddings
```

Or manually in your `Cargo.toml`:

```toml
[dependencies]
rusty-embeddings = { git = "https://github.com/sindre0830/rusty-embeddings.git", tag = "v1.0.0" }
```

### Example

---

## API Overview

### Core Functions

---

## Development Guide

### Prerequisites

### Commands

| Command            | Description                             | Example                                                    |
| ------------------ | --------------------------------------- | ---------------------------------------------------------- |
| **Build**          | Compiles the crate in release mode      | `cargo build --release`                                    |
| **Run Tests**      | Executes all unit tests                 | `cargo test`                                               |
| **Lint (Clippy)**  | Checks for style and performance issues | `cargo clippy --all-targets --all-features -- -D warnings` |
| **Format Code**    | Formats the entire codebase             | `cargo fmt`                                                |
| **Doc Generation** | Builds local documentation              | `cargo doc --open`                                         |

---

### Upgrading Dependencies

To upgrade all dependencies to the latest compatible versions:

```bash
cargo update
```

Or for a specific crate:

```bash
cargo update -p crate-name
```
