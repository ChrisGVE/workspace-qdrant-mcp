# N Languages LSP Research Report

## Executive Summary

This comprehensive report analyzes all programming languages beginning with "N" from Wikipedia's List of Programming Languages, focusing on Language Server Protocol (LSP) implementations and modern development tooling. Of the 21 languages analyzed, 4 languages have mature LSP ecosystems (Nim, Nix, NASM/Assembly, Nu), 1 has partial LSP support (NWScript), and 16 languages lack viable LSP implementations.

**Key Findings:**
- **Primary Recommendations:** Nim, Nix, Nu (Nushell)
- **Secondary Options:** NASM (via asm-lsp), NWScript
- **Languages with Strong Ecosystems:** Nim and Nix lead with multiple LSP server options
- **Emerging Technologies:** Nu shell includes built-in LSP server
- **Notable Gaps:** NetLogo, NewLISP, and NetRexx would benefit from community LSP development

## Detailed Language Analysis

### 1. **Nim** ⭐⭐⭐⭐⭐
**Status:** Primary Choice - Mature LSP Ecosystem

**File Signatures:**
- Extensions: `.nim`, `.nims` (NimScript), `.nimble` (package files)
- Shebang: `#!/usr/bin/env nim`
- Content signature: `proc`, `var`, `let`, `type`, `import`
- Source: https://nim-lang.org/

**LSP Servers:**
1. **nimlsp** (Primary)
   - Repository: https://github.com/PMunch/nimlsp
   - Official Status: Community-maintained, widely adopted
   - LSP Compliance: Full LSP 3.17 support
   - Features: Auto-completion, go-to-definition, hover, diagnostics

2. **nimlangserver** (Alternative)
   - Repository: https://github.com/nim-lang/langserver
   - Official Status: Official Nim organization
   - LSP Compliance: Full LSP support
   - Features: External nimsuggest process for stability

**Installation Instructions:**
```bash
# Primary installation via nimble
nimble install nimlsp

# macOS via Homebrew
brew install nim
nimble install nimlsp

# Linux via package managers
apt install nim         # Ubuntu/Debian
pacman -S nim          # Arch Linux
nimble install nimlsp

# Windows via Scoop
scoop install nim
nimble install nimlsp

# Manual compilation
git clone https://github.com/PMunch/nimlsp
cd nimlsp
nim c -d:explicitSourcePath=/path/to/nim/sources nimlsp
```

**Tree-sitter Grammar:** ✅ Multiple implementations
- https://github.com/alaviss/tree-sitter-nim (most complete)
- https://github.com/aMOPel/tree-sitter-nim
- https://github.com/monaqa/tree-sitter-nim

**Evaluation:**
- Speed: ⭐⭐⭐⭐⭐ (Excellent compilation speed, efficient runtime)
- Project Scope: ⭐⭐⭐⭐⭐ (Systems programming to web development)
- Feature Completeness: ⭐⭐⭐⭐⭐ (Full LSP features, multiple servers)
- Active Development: ⭐⭐⭐⭐⭐ (Very active language and tooling development)
- LSP Standard Compliance: ⭐⭐⭐⭐⭐ (Full LSP 3.17 support)

### 2. **Nix** ⭐⭐⭐⭐⭐
**Status:** Primary Choice - Multiple Mature LSP Options

**File Signatures:**
- Extensions: `.nix`
- Content signature: `{ }`, `let`, `in`, `with`, `import`, `rec`
- Flake files: `flake.nix`
- Source: https://nixos.org/

**LSP Servers:**
1. **nil** (Recommended)
   - Repository: https://github.com/oxalica/nil
   - Official Status: Community-maintained, most popular
   - LSP Compliance: Full LSP support
   - Features: Incremental analysis, completions, diagnostics

2. **nixd** (Advanced)
   - Repository: https://github.com/nix-community/nixd
   - Official Status: Community-maintained, active development
   - LSP Compliance: Full LSP support with advanced features
   - Features: NixOS options completion, official Nix library integration

3. **rnix-lsp** (Legacy)
   - Repository: https://github.com/nix-community/rnix-lsp
   - Official Status: Maintenance mode
   - LSP Compliance: Basic LSP support

**Installation Instructions:**
```bash
# nil installation
nix-env -i nil
# or
nix profile install nixpkgs#nil

# nixd installation
nix-env -i nixd
# or
nix profile install nixpkgs#nixd

# rnix-lsp (legacy)
nix-env -i -f https://github.com/nix-community/rnix-lsp/archive/master.tar.gz

# macOS via Homebrew
brew install nil
brew install nixd

# Using flakes
nix shell nixpkgs#nil
nix shell nixpkgs#nixd
```

**Tree-sitter Grammar:** ✅ Available
- Part of standard tree-sitter grammar collections
- https://github.com/ratson/nix-treesitter

**Evaluation:**
- Speed: ⭐⭐⭐⭐ (Good performance, some complex evaluations can be slow)
- Project Scope: ⭐⭐⭐⭐⭐ (Package management to full system configuration)
- Feature Completeness: ⭐⭐⭐⭐⭐ (Multiple servers with comprehensive features)
- Active Development: ⭐⭐⭐⭐⭐ (Very active with multiple competing implementations)
- LSP Standard Compliance: ⭐⭐⭐⭐⭐ (Full compliance across multiple servers)

### 3. **Nu (Nushell)** ⭐⭐⭐⭐⭐
**Status:** Primary Choice - Built-in LSP Server

**File Signatures:**
- Extensions: `.nu`
- Shebang: `#!/usr/bin/env nu`
- Content signature: Shell commands with structured data pipes
- Source: https://www.nushell.sh/

**LSP Server:**
- **Built-in LSP** (Official)
  - Repository: https://github.com/nushell/nushell
  - Official Status: Built into Nushell core
  - LSP Compliance: Growing LSP support
  - Features: Diagnostics, completions, syntax highlighting

**Installation Instructions:**
```bash
# Install Nushell (includes LSP)
# macOS via Homebrew
brew install nushell

# Linux via package managers
apt install nushell      # Ubuntu/Debian (recent versions)
pacman -S nushell       # Arch Linux
dnf install nushell     # Fedora

# Windows via Scoop
scoop install nu

# Cargo installation
cargo install nu

# Start LSP server
nu --lsp
```

**Tree-sitter Grammar:** ✅ Official implementation
- https://github.com/nushell/tree-sitter-nu

**Evaluation:**
- Speed: ⭐⭐⭐⭐ (Fast shell operations, some complex parsing slower)
- Project Scope: ⭐⭐⭐⭐ (Modern shell with structured data focus)
- Feature Completeness: ⭐⭐⭐⭐ (Built-in LSP, actively expanding features)
- Active Development: ⭐⭐⭐⭐⭐ (Very active development, regular releases)
- LSP Standard Compliance: ⭐⭐⭐⭐ (Good compliance, improving with each release)

### 4. **NASM (Assembly)** ⭐⭐⭐⭐
**Status:** Primary Choice - Strong Assembly LSP Support

**File Signatures:**
- Extensions: `.asm`, `.s`, `.S`, `.nasm`
- Content signature: Assembly mnemonics, `section`, `global`, `extern`
- Source: https://www.nasm.us/

**LSP Server:**
- **asm-lsp** (Primary)
  - Repository: https://github.com/bergercookie/asm-lsp
  - Official Status: Community-maintained, widely adopted
  - LSP Compliance: Full LSP support
  - Features: Multiple assemblers (NASM, GAS, MASM), multiple architectures

**Installation Instructions:**
```bash
# Cargo installation
cargo install asm-lsp
# or from GitHub
cargo install --git https://github.com/bergercookie/asm-lsp asm-lsp

# Configuration generation
asm-lsp gen-config

# Editor configuration required for .asm-lsp.toml project config
```

**Tree-sitter Grammar:** ✅ Available
- https://github.com/naclsn/tree-sitter-nasm

**Evaluation:**
- Speed: ⭐⭐⭐⭐⭐ (Excellent performance for assembly analysis)
- Project Scope: ⭐⭐⭐ (Assembly language development, systems programming)
- Feature Completeness: ⭐⭐⭐⭐ (Good LSP features, multiple architecture support)
- Active Development: ⭐⭐⭐⭐ (Active development, regular updates)
- LSP Standard Compliance: ⭐⭐⭐⭐ (Good LSP compliance)

### 5. **NWScript** ⭐⭐⭐
**Status:** Secondary Option - Game-Specific LSP Support

**File Signatures:**
- Extensions: `.nss`, `.ncs` (compiled)
- Content signature: C-like syntax with game-specific functions
- Source: Neverwinter Nights scripting language

**LSP Server:**
- **nwscript-ee-language-server**
  - Repository: https://github.com/PhilippeChab/nwscript-ee-language-server
  - Official Status: Community-maintained
  - LSP Compliance: Basic LSP support
  - Features: Diagnostics via nwnsc compiler, VS Code extension

**Installation Instructions:**
```bash
# VS Code extension installation
# Search for "NWScript" in VS Code marketplace

# Manual setup requires nwnsc compiler
# Download from: https://github.com/nwneetools/nwnsc
```

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: ⭐⭐⭐ (Adequate for game scripting)
- Project Scope: ⭐⭐ (Limited to Neverwinter Nights modding)
- Feature Completeness: ⭐⭐⭐ (Basic LSP features, compilation diagnostics)
- Active Development: ⭐⭐⭐ (Community maintenance, limited scope)
- LSP Standard Compliance: ⭐⭐⭐ (Basic compliance)

## Languages Without Viable LSP Support

### 6. **NetLogo** ⭐⭐
**Status:** No Viable LSP - Syntax Highlighting Only

**File Signatures:**
- Extensions: `.nlogo`, `.nls`
- Content signature: Logo-style commands, `to`, `end`, `ask`, `breed`
- Source: https://ccl.northwestern.edu/netlogo/

**Available Tooling:**
- VS Code syntax highlighting extension
- Built-in NetLogo IDE
- No LSP server implementation

**Tree-sitter Grammar:** ❌ Not available

### 7. **NewLISP** ⭐⭐
**Status:** No Viable LSP - Minimal Tooling

**File Signatures:**
- Extensions: `.lsp`, `.cgi`
- Content signature: Lisp syntax with `(define`, `(lambda`, `(if`
- Source: http://www.newlisp.org/

**Available Tooling:**
- Built-in development environment (newLISP-GS)
- Basic syntax highlighting in some editors
- No LSP server implementation

**Tree-sitter Grammar:** ❌ Not available

### 8. **Neko** ⭐⭐
**Status:** No Viable LSP - Legacy Haxe Target

**File Signatures:**
- Extensions: `.neko`, `.n` (bytecode)
- Content signature: Haxe-style syntax or native Neko syntax
- Source: https://nekovm.org/

**Available Tooling:**
- Part of Haxe toolchain
- Limited standalone tooling
- No dedicated LSP server

**Tree-sitter Grammar:** ❌ Not available

### 9. **Nemerle** ⭐⭐
**Status:** No Viable LSP - .NET Language Without Modern Tooling

**File Signatures:**
- Extensions: `.n`
- Content signature: C#-like syntax with macros, `.NET` integration
- Source: Historical .NET language project

**Available Tooling:**
- Visual Studio integration (legacy)
- No modern LSP implementation

**Tree-sitter Grammar:** ❌ Not available

### 10. **NetRexx** ⭐⭐
**Status:** No Viable LSP - Legacy IBM Language

**File Signatures:**
- Extensions: `.nrx`
- Content signature: REXX syntax with Java integration
- Source: https://www.netrexx.org/

**Available Tooling:**
- Basic Rexx editor support
- No NetRexx-specific LSP server

**Tree-sitter Grammar:** ❌ Not available

### 11. **NSIS** ⭐⭐
**Status:** No Viable LSP - Install Script Language

**File Signatures:**
- Extensions: `.nsi`, `.nsh`
- Content signature: Install script commands, `Section`, `Function`
- Source: https://nsis.sourceforge.io/

**Available Tooling:**
- VS Code syntax highlighting extensions
- HM NIS Edit IDE
- No LSP server implementation

**Tree-sitter Grammar:** ❌ Not available

## Minor/Legacy Languages (Limited Analysis)

The following languages were identified but have minimal contemporary relevance or tooling:

12. **Napier88** - Academic language, no modern tooling
13. **NESL** - Parallel programming research language
14. **Net.Data** - Legacy IBM web scripting
15. **NEWP** - Historical programming language
16. **Newspeak** - Smalltalk-inspired research language
17. **NewtonScript** - Legacy Apple Newton scripting
18. **Nial** - Array programming language
19. **Nord Programming Language (NPL)** - Specialized domain language
20. **Not eXactly C (NXC)** - LEGO Mindstorms programming
21. **Not Quite C (NQC)** - Predecessor to NXC

## Recommendations by Use Case

### **Systems Programming & Performance**
1. **Nim** - Excellent performance, Python-like syntax, growing ecosystem
2. **NASM** - When assembly language is required

### **Configuration Management & DevOps**
1. **Nix** - Powerful package/system management with multiple LSP options
2. **Nu** - Modern shell with structured data handling

### **Shell Scripting & Command Line**
1. **Nu (Nushell)** - Modern shell with built-in LSP support
2. **Nix** - For complex system configurations

### **Game Development & Modding**
1. **NWScript** - Only viable option for Neverwinter Nights modding

### **Educational & Research**
1. **NetLogo** - Agent-based modeling (limited tooling)
2. **Nim** - Beginner-friendly systems language

## Conclusion

The "N" languages present a diverse landscape with clear leaders in LSP support. **Nim**, **Nix**, and **Nu** represent the best choices for modern development with comprehensive tooling ecosystems. **NASM** serves specialized assembly development needs well. Most other "N" languages lack viable LSP implementations, representing opportunities for community development or indicating their specialized/legacy status.

The analysis reveals that LSP adoption correlates strongly with language vitality and community engagement, with actively developed languages providing multiple LSP server options while legacy languages often lack any modern tooling support.