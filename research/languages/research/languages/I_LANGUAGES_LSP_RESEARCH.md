# I Languages LSP Research Report

## Executive Summary

This comprehensive analysis covers all programming languages starting with "I" from Wikipedia's List of Programming Languages, evaluating their Language Server Protocol (LSP) support, installation methods, and development ecosystem maturity. Of the 10 primary languages analyzed, only **Idris** and **Inform** have viable LSP implementations, while **IDL** has excellent VS Code extension support but no traditional LSP server.

### Key Findings:
- **2 languages** with full LSP support (Idris, Inform 6)
- **1 language** with advanced editor support but no LSP (IDL - VS Code extension)
- **7 languages** without viable LSP implementations
- **Tree-sitter support** available for IDL and Idris only

## Detailed Language Analysis

### 1. IDL (Interactive Data Language) ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.pro`
- Content signatures: IDL source code with scientific computing functions
- Shebangs: None (compiled language)

**Editor Support:**
- **Official IDL Extension for VS Code**: [marketplace.visualstudio.com/items?itemName=IDL.idl-for-vscode](https://marketplace.visualstudio.com/items?itemName=IDL.idl-for-vscode)
- Repository: [github.com/interactive-data-language/vscode-idl](https://github.com/interactive-data-language/vscode-idl)
- Status: Official, actively maintained
- Features: Syntax highlighting, code snippets, variable type detection, auto-complete, debugging

**Installation:**
```bash
# VS Code Extension
code --install-extension IDL.idl-for-vscode

# Requirements: IDL 8.8.0+ or ENVI 6.0/5.7
```

**LSP Compliance:** ❌ No traditional LSP server, but advanced VS Code extension with language features

**Evaluation:**
- Speed: ⭐⭐⭐⭐⭐ (Excellent performance)
- Project Scope: ⭐⭐⭐⭐⭐ (Scientific computing, widely used)
- Feature Completeness: ⭐⭐⭐⭐⭐ (Full IDE features)
- Active Development: ⭐⭐⭐⭐⭐ (Official support)
- LSP Standard Compliance: ⭐⭐ (Custom extension, not LSP)

**Tree-sitter Grammar:** ✅ Available ([github.com/cathaysia/tree-sitter-idl](https://github.com/cathaysia/tree-sitter-idl))

**Recommendation:** **Primary Choice** for IDL development via VS Code extension

---

### 2. Idris ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.idr`
- Content signatures: Dependent type annotations, pure functional syntax
- Shebangs: None

**LSP Server:**
- **Name**: idris2-lsp
- Repository: [github.com/idris-community/idris2-lsp](https://github.com/idris-community/idris2-lsp)
- Status: Community-maintained, active development
- LSP Compliance: ✅ Full LSP implementation

**Installation:**
```bash
# Prerequisites: Idris2 compiler required
git clone https://github.com/idris-community/idris2-lsp.git
cd idris2-lsp
git submodule update --init Idris2
cd Idris2 && make bootstrap SCHEME=chez && make install
cd .. && make install

# VS Code Extension
code --install-extension bamboo.idris2-lsp

# Emacs via lsp-mode (built-in support)
```

**Evaluation:**
- Speed: ⭐⭐⭐⭐ (Good performance)
- Project Scope: ⭐⭐⭐ (Academic/research use)
- Feature Completeness: ⭐⭐⭐⭐ (Good LSP features)
- Active Development: ⭐⭐⭐⭐ (Community active)
- LSP Standard Compliance: ⭐⭐⭐⭐⭐ (Full compliance)

**Tree-sitter Grammar:** ✅ Available ([github.com/gwerbin/tree-sitter-idris2](https://github.com/gwerbin/tree-sitter-idris2))

**Recommendation:** **Primary Choice** for functional programming with dependent types

---

### 3. Inform (Interactive Fiction) ⭐⭐⭐

**File Signatures:**
- Extensions: `.inf` (Inform 6), `.ni` (Inform 7), `.i7x` (extensions)
- Content signatures: Natural language programming syntax
- Shebangs: None

**LSP Server:**
- **Name**: ls4inform6
- Repository: [github.com/toerob/ls4inform6](https://github.com/toerob/ls4inform6)
- Status: Community-maintained (Inform 6 only)
- LSP Compliance: ✅ XText-powered LSP server

**Installation:**
```bash
# Eclipse plugin (requires Eclipse 2020-03+)
# Install from Eclipse Marketplace

# VS Code support via LSP
# Manual installation from GitHub releases
```

**Evaluation:**
- Speed: ⭐⭐⭐ (Adequate performance)
- Project Scope: ⭐⭐⭐ (Interactive fiction niche)
- Feature Completeness: ⭐⭐⭐ (Basic LSP features)
- Active Development: ⭐⭐ (Limited activity)
- LSP Standard Compliance: ⭐⭐⭐⭐ (Good compliance)

**Tree-sitter Grammar:** ❌ Not available

**Recommendation:** **Secondary Option** for Inform 6 development only

---

### 4. Io ⭐⭐

**File Signatures:**
- Extensions: `.io`
- Content signatures: Prototype-based object syntax, message passing
- Shebangs: `#!/usr/bin/env io`

**LSP Server:** ❌ None available

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐⭐ (Niche prototype-based language)
- Feature Completeness: N/A
- Active Development: ⭐ (Minimal activity)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Basic text editor support only

---

### 5. Icon ⭐⭐

**File Signatures:**
- Extensions: `.icn`
- Content signatures: Goal-directed execution syntax, string processing
- Shebangs: None

**LSP Server:** ❌ None available

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐⭐ (Academic/legacy)
- Feature Completeness: N/A
- Active Development: ⭐ (Minimal activity)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Basic text editor support only

---

### 6. ISLISP ⭐

**File Signatures:**
- Extensions: `.lsp`, `.lisp`
- Content signatures: S-expression syntax, Lisp dialect
- Shebangs: None

**LSP Server:** ❌ None available (related Emacs Lisp support exists)

**Tree-sitter Grammar:** ❌ Not available (Emacs Lisp grammar available)

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐ (Very limited use)
- Feature Completeness: N/A
- Active Development: ⭐ (Minimal activity)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Consider Common Lisp alternatives

---

### 7. IBM RPG ⭐⭐

**File Signatures:**
- Extensions: `.rpg`, `.rpgle`
- Content signatures: Business application syntax, IBM i specific
- Shebangs: None

**LSP Server:** ⚠️ Specialized implementations exist
- Custom LSP with RPG-to-Python translation
- IBM i-specific tooling

**Installation:**
```bash
# VS Code Extension
code --install-extension barrettotte.ibmi-languages
```

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: ⭐⭐ (Adequate for specialized use)
- Project Scope: ⭐⭐⭐ (Enterprise IBM i)
- Feature Completeness: ⭐⭐ (Basic features)
- Active Development: ⭐⭐ (IBM ecosystem)
- LSP Standard Compliance: ⭐⭐ (Non-standard implementations)

**Recommendation:** **Secondary Option** for IBM i development only

---

### 8. IBM Basic Assembly Language ⭐

**File Signatures:**
- Extensions: `.bal`, `.asm`, `.hlasm`
- Content signatures: Mainframe assembly instructions
- Shebangs: None

**LSP Server:** ❌ None available (IBM Z Open Editor supports higher-level languages only)

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐ (Legacy mainframe)
- Feature Completeness: N/A
- Active Development: ⭐ (Legacy)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Use traditional mainframe tools

---

### 9. IBM Informix-4GL ⭐

**File Signatures:**
- Extensions: `.4gl`
- Content signatures: Database 4GL syntax, embedded SQL
- Shebangs: None

**LSP Server:** ❌ None available

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐ (Legacy database applications)
- Feature Completeness: N/A
- Active Development: ⭐ (Legacy)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Use IBM Informix tools

---

### 10. Instruction List ⭐

**File Signatures:**
- Extensions: Various (IEC 61131-3 dependent)
- Content signatures: PLC assembly-like instructions
- Shebangs: None

**LSP Server:** ❌ None available (deprecated in IEC 61131-3:2025)

**Tree-sitter Grammar:** ❌ Not available

**Evaluation:**
- Speed: N/A
- Project Scope: ⭐ (Deprecated PLC language)
- Feature Completeness: N/A
- Active Development: ⭐ (Deprecated)
- LSP Standard Compliance: N/A

**Recommendation:** **No Viable LSP** - Use modern IEC 61131-3 alternatives

## Recommendations by Use Case

### Scientific Computing
**Primary**: IDL with official VS Code extension
- Comprehensive features for scientific data analysis
- Official support from NV5 Geospatial
- Rich debugging and visualization capabilities

### Functional Programming Research
**Primary**: Idris with idris2-lsp
- Modern dependent types system
- Active academic community
- Full LSP compliance with good tooling

### Interactive Fiction Development
**Secondary**: Inform 6 with ls4inform6
- Specialized LSP for Inform 6 only
- Limited but functional IDE features
- Consider Inform 7 with manual tools

### Enterprise/Legacy Development
**Secondary**: IBM RPG with specialized tooling
- Limited to IBM i environments
- Non-standard LSP implementations
- Consider migration to modern languages

### All Other Languages
**None**: No viable LSP options available
- Use basic syntax highlighting where available
- Consider language migration for modern development
- Rely on traditional development tools

## Installation Priority Matrix

| Language | LSP Quality | Installation Difficulty | Recommendation |
|----------|-------------|------------------------|----------------|
| IDL      | High (Non-LSP) | Easy | Primary Choice |
| Idris    | High | Medium | Primary Choice |
| Inform   | Medium | Hard | Secondary Option |
| IBM RPG  | Low | Medium | Secondary Option |
| Others   | None | N/A | No Viable LSP |

## Future Outlook

The "I" language ecosystem shows limited LSP adoption, with only functional/academic languages (Idris) and specialized domains (IDL, Inform) having meaningful tooling support. Legacy enterprise languages (IBM family) rely on proprietary tooling, while esoteric languages (Io, Icon) have minimal modern development support.

**Recommendation**: Focus development efforts on IDL and Idris for their respective domains, consider migration paths for legacy languages, and use traditional tooling for the remaining languages until community-driven LSP implementations emerge.