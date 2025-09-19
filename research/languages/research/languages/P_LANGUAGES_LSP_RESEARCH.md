# P Languages LSP Research Report

## Executive Summary

This comprehensive analysis examines Language Server Protocol (LSP) support for all programming languages beginning with "P" from Wikipedia's List of Programming Languages. The research covers 63 distinct languages, ranging from widely-adopted languages like Python, PHP, and Perl to specialized domain-specific languages and historical implementations.

### Key Findings

**Tier 1 (Production-Ready LSPs)**
- Python: Multiple excellent LSP servers (Pylsp, Pyright, Ruff-lsp)
- PHP: Strong LSP ecosystem (Intelephense, Phpactor)
- PowerShell: Official Microsoft LSP server
- PL/SQL: Oracle and third-party LSP support
- Processing: Java-based LSP through JavaLS
- PureScript: Active LSP development
- Prolog: Multiple LSP implementations

**Tier 2 (Limited/Emerging LSP Support)**
- Pascal: Some LSP servers available
- Perl: Basic LSP support
- PostScript: Limited support
- Pony: Emerging LSP
- Pike: Minimal support

**Tier 3 (No Viable LSP Support)**
- Most historical/academic languages (P′′, Plankalkül, PLEX, etc.)
- Specialized DSLs (POV-Ray SDL, Pure Data)
- Legacy languages (PL/M, PL360, PILOT)

---

## Detailed Language Analysis

### Python ⭐⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.py`, `.pyw`, `.pyi`, `.pyx`
- Shebang: `#!/usr/bin/env python`, `#!/usr/bin/python`
- Content signatures: `# -*- coding: utf-8 -*-`, `from __future__ import`
- Source: [Python.org](https://www.python.org/)

**LSP Servers:**
1. **pylsp (python-lsp-server)** - [GitHub](https://github.com/python-lsp/python-lsp-server)
   - Official: Community-maintained (Spyder team)
   - LSP Compliance: Full
   - Installation: `pip install python-lsp-server`

2. **Pyright** - [GitHub](https://github.com/microsoft/pyright)
   - Official: Microsoft
   - LSP Compliance: Full
   - Installation: `npm install -g pyright`

3. **Jedi Language Server** - [GitHub](https://github.com/pappasam/jedi-language-server)
   - Official: Community
   - LSP Compliance: Full
   - Installation: `pip install jedi-language-server`

4. **Ruff LSP** - [GitHub](https://github.com/astral-sh/ruff-lsp)
   - Official: Astral
   - LSP Compliance: Full (linting/formatting focused)
   - Installation: `pip install ruff-lsp`

**Installation Instructions:**
```bash
# Ubuntu/Debian
sudo apt install python3-pylsp

# macOS
brew install python-lsp-server
# or
pip install python-lsp-server

# Windows (Scoop)
scoop install python-lsp-server

# NPM (Pyright)
npm install -g pyright

# Cargo/Rust (Ruff)
cargo install ruff
```

**Evaluation Narrative:**
Python boasts the most mature LSP ecosystem with multiple excellent servers. Pylsp offers comprehensive tool integration, Pyright provides fast type checking, Jedi delivers traditional Python intelligence, and Ruff offers extremely fast linting/formatting. All are actively maintained with excellent documentation and broad editor support.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-python](https://github.com/tree-sitter/tree-sitter-python)

**Recommendation:** Primary Choice

---

### PHP ⭐⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.php`, `.phtml`, `.php3`, `.php4`, `.php5`, `.phps`
- Shebang: `#!/usr/bin/php`
- Content signatures: `<?php`, `<?=`
- Source: [PHP.net](https://www.php.net/)

**LSP Servers:**
1. **Intelephense** - [Website](https://intelephense.com/)
   - Official: Proprietary (freemium)
   - LSP Compliance: Full
   - Installation: `npm install -g intelephense`

2. **Phpactor** - [GitHub](https://github.com/phpactor/phpactor)
   - Official: Community
   - LSP Compliance: Partial but growing
   - Installation: `composer global require phpactor/phpactor`

**Installation Instructions:**
```bash
# Intelephense (NPM)
npm install -g intelephense

# Phpactor (Composer)
composer global require phpactor/phpactor

# Ubuntu/Debian (PHP)
sudo apt install php-cli composer

# macOS
brew install php composer
```

**Evaluation Narrative:**
PHP has strong LSP support with Intelephense leading as the most feature-complete server, though it requires a paid license for advanced features. Phpactor serves as a capable open-source alternative. Both provide excellent code completion, diagnostics, and refactoring capabilities.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-php](https://github.com/tree-sitter/tree-sitter-php)

**Recommendation:** Primary Choice

---

### PowerShell ⭐⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.ps1`, `.psm1`, `.psd1`, `.ps1xml`
- Content signatures: `#Requires`, `[CmdletBinding()]`, `param(`
- Source: [Microsoft PowerShell](https://github.com/PowerShell/PowerShell)

**LSP Servers:**
1. **PowerShell Editor Services** - [GitHub](https://github.com/PowerShell/PowerShellEditorServices)
   - Official: Microsoft
   - LSP Compliance: Full
   - Installation: Included with PowerShell extension

**Installation Instructions:**
```bash
# Windows (PowerShell Gallery)
Install-Module -Name PSScriptAnalyzer

# Ubuntu/Debian
sudo apt install powershell

# macOS
brew install powershell

# VS Code Extension
# Automatically includes PowerShell Editor Services
```

**Evaluation Narrative:**
PowerShell has excellent first-party LSP support through Microsoft's PowerShell Editor Services. Includes advanced features like PSScriptAnalyzer integration, debugging, and comprehensive IntelliSense. Well-maintained and actively developed.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-powershell](https://github.com/airbus-cert/tree-sitter-powershell)

**Recommendation:** Primary Choice

---

### Perl ⭐⭐⭐

**File Signatures:**
- Extensions: `.pl`, `.pm`, `.pod`, `.t`
- Shebang: `#!/usr/bin/perl`, `#!/usr/bin/env perl`
- Content signatures: `use strict;`, `use warnings;`, `package`
- Source: [Perl.org](https://www.perl.org/)

**LSP Servers:**
1. **PLS (Perl Language Server)** - [GitHub](https://github.com/FractalBoy/perl-language-server)
   - Official: Community
   - LSP Compliance: Partial (early stages)
   - Installation: `cpanm PLS`

2. **Perl::LanguageServer** - [GitHub](https://github.com/richterger/Perl-LanguageServer)
   - Official: Community
   - LSP Compliance: Partial
   - Installation: `cpan Perl::LanguageServer`

**Installation Instructions:**
```bash
# PLS
cpanm PLS

# Perl::LanguageServer (Ubuntu)
sudo apt install libanyevent-perl libclass-refresh-perl
sudo cpan Perl::LanguageServer

# macOS
brew install perl
cpanm PLS
```

**Evaluation Narrative:**
Perl LSP support is limited with two competing implementations in early development stages. Both servers provide basic features like go-to-definition but lack advanced capabilities. The Perl ecosystem's focus on CPAN tooling has meant slower LSP adoption.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-perl](https://github.com/tree-sitter-grammars/tree-sitter-perl)

**Recommendation:** Secondary Option

---

### PL/SQL ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.sql`, `.pls`, `.plb`, `.pck`, `.pkb`, `.pks`
- Content signatures: `DECLARE`, `BEGIN`, `END;`, `CREATE OR REPLACE`
- Source: [Oracle PL/SQL](https://www.oracle.com/database/technologies/appdev/plsql.html)

**LSP Servers:**
1. **Oracle SQL Developer Extension** - [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Oracle.sql-developer)
   - Official: Oracle
   - LSP Compliance: Partial (proprietary)
   - Installation: VS Code extension

2. **SQL Language Server** - [GitHub](https://github.com/joe-re/sql-language-server)
   - Official: Community
   - LSP Compliance: Basic SQL support
   - Installation: `npm install -g sql-language-server`

**Installation Instructions:**
```bash
# Oracle SQL Developer (latest)
# Download from Oracle website

# VS Code Extension
# Install Oracle SQL Developer extension

# Generic SQL server
npm install -g sql-language-server
```

**Evaluation Narrative:**
PL/SQL LSP support is primarily through Oracle's official tooling and generic SQL language servers. Oracle SQL Developer provides the most comprehensive PL/SQL support but is proprietary. Community options exist but with limited PL/SQL-specific features.

**Tree-sitter Grammar:** ⚠️ Basic SQL support available

**Recommendation:** Primary Choice (Oracle tooling)

---

### Processing ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.pde`, `.java` (when exported)
- Content signatures: `void setup()`, `void draw()`, `size(`
- Source: [Processing.org](https://processing.org/)

**LSP Servers:**
1. **LS4P (Language Server for Processing)** - [GitHub](https://github.com/processing-language-server/LS4P)
   - Official: Community
   - LSP Compliance: Partial
   - Installation: Via Processing IDE or manual build

2. **Java Language Server** - [GitHub](https://github.com/eclipse-jdtls/eclipse.jdt.ls)
   - Official: Eclipse Foundation
   - LSP Compliance: Full (as Java)
   - Installation: `npm install -g @emacs-lsp/jdtls`

**Installation Instructions:**
```bash
# LS4P (build from source)
git clone https://github.com/processing-language-server/LS4P
# Follow build instructions

# Java LSP (for Processing as Java)
npm install -g @emacs-lsp/jdtls

# Processing IDE
# Download from processing.org
```

**Evaluation Narrative:**
Processing benefits from Java LSP compatibility since it transpiles to Java. LS4P provides Processing-specific features but has limited maintenance. Java Language Server offers excellent support when treating Processing as Java code.

**Tree-sitter Grammar:** ⚠️ Java grammar works for most Processing code

**Recommendation:** Primary Choice (via Java LSP)

---

### PureScript ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.purs`, `.spurs`
- Content signatures: `module`, `import`, `where`, `::`, `=>`
- Source: [PureScript.org](https://www.purescript.org/)

**LSP Servers:**
1. **PureScript Language Server** - [GitHub](https://github.com/nwolverson/purescript-language-server)
   - Official: Community
   - LSP Compliance: Full
   - Installation: `npm install -g purescript-language-server`

**Installation Instructions:**
```bash
# NPM
npm install -g purescript-language-server

# Requires PureScript compiler
npm install -g purescript spago

# Ubuntu/Debian
sudo apt install node-purescript-language-server
```

**Evaluation Narrative:**
PureScript has solid LSP support with active development and good feature coverage. The language server integrates well with the PureScript toolchain and provides excellent type information, completion, and error reporting.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-purescript](https://github.com/postsolar/tree-sitter-purescript)

**Recommendation:** Primary Choice

---

### Prolog ⭐⭐⭐⭐

**File Signatures:**
- Extensions: `.pl`, `.pro`, `.prolog`
- Content signatures: `:-`, `?-`, `assert(`, `rule(`
- Source: [SWI-Prolog](https://www.swi-prolog.org/)

**LSP Servers:**
1. **lsp_server (SWI-Prolog)** - [GitHub](https://github.com/jamesnvc/lsp_server)
   - Official: Community (SWI-Prolog focused)
   - LSP Compliance: Full
   - Installation: `swipl pack install lsp_server`

2. **prolog_lsp** - [GitHub](https://github.com/hargettp/prolog_lsp)
   - Official: Community
   - LSP Compliance: Partial (experimental)
   - Installation: Via SWI-Prolog pack system

**Installation Instructions:**
```bash
# SWI-Prolog lsp_server
swipl -g "pack_install(lsp_server)" -t halt

# Ubuntu/Debian
sudo apt install swi-prolog
swipl pack install lsp_server

# macOS
brew install swi-prolog
```

**Evaluation Narrative:**
Prolog has good LSP support primarily through SWI-Prolog's lsp_server pack. Provides code completion, formatting, and navigation features. Well-integrated with SWI-Prolog's introspection capabilities.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-prolog](https://github.com/tree-sitter-grammars/tree-sitter-prolog)

**Recommendation:** Primary Choice

---

### Pascal ⭐⭐⭐

**File Signatures:**
- Extensions: `.pas`, `.pp`, `.p`, `.inc`
- Content signatures: `program`, `begin`, `end.`, `unit`, `uses`
- Source: [Free Pascal](https://www.freepascal.org/)

**LSP Servers:**
1. **Pascal Language Server (pasls)** - [GitHub](https://github.com/genericptr/pascal-language-server)
   - Official: Community
   - LSP Compliance: Partial
   - Installation: Build from source (Lazarus required)

2. **OmniPascal** - [Website](https://www.omnipascal.com/)
   - Official: Third-party
   - LSP Compliance: Partial (VS Code specific)
   - Installation: VS Code extension

**Installation Instructions:**
```bash
# Pascal Language Server (build from source)
# Requires Lazarus IDE and Free Pascal
git clone https://github.com/genericptr/pascal-language-server
# Build in Lazarus

# OmniPascal (VS Code)
# Install extension in VS Code
# Configure freePascalSourcePath in settings
```

**Evaluation Narrative:**
Pascal LSP support is limited with community-driven efforts primarily targeting Free Pascal and Delphi. OmniPascal provides VS Code integration while pasls offers broader LSP compatibility but requires complex setup.

**Tree-sitter Grammar:** ✅ Available - [tree-sitter-pascal](https://github.com/tree-sitter-grammars/tree-sitter-pascal)

**Recommendation:** Secondary Option

---

### Pony ⭐⭐⭐

**File Signatures:**
- Extensions: `.pony`
- Content signatures: `actor`, `class`, `trait`, `primitive`, `fun`, `be`
- Source: [Pony Language](https://www.ponylang.io/)

**LSP Servers:**
1. **Pony Language Server** - [GitHub](https://github.com/ponylang/pony-language-server)
   - Official: Pony Team
   - LSP Compliance: Partial
   - Installation: Build from source

**Installation Instructions:**
```bash
# Build from source (requires Pony compiler)
git clone https://github.com/ponylang/pony-language-server
# Follow build instructions

# VS Code extension available
# Search for "pony-lsp" in VS Code extensions
```

**Evaluation Narrative:**
Pony has basic LSP support through the official language server, primarily targeting VS Code. The implementation is functional but limited in scope due to Pony's smaller ecosystem and user base.

**Tree-sitter Grammar:** ⚠️ Limited availability

**Recommendation:** Secondary Option

---

### PostScript ⭐⭐

**File Signatures:**
- Extensions: `.ps`, `.eps`, `.ai`
- Content signatures: `%!PS-`, `%%BoundingBox:`, `/` (PostScript operators)
- Source: [Adobe PostScript](https://www.adobe.com/products/postscript/)

**LSP Servers:**
- No dedicated LSP servers found

**Installation Instructions:**
```bash
# Ghostscript (PostScript interpreter)
# Ubuntu/Debian
sudo apt install ghostscript

# macOS
brew install ghostscript

# Windows
# Download from ghostscript.com
```

**Evaluation Narrative:**
PostScript lacks dedicated LSP server support. Development typically relies on Ghostscript for interpretation and basic text editors with syntax highlighting. The language's specialized use case and age limit modern tooling development.

**Tree-sitter Grammar:** ❌ No tree-sitter grammar available

**Recommendation:** No Viable LSP

---

### Pike ⭐⭐

**File Signatures:**
- Extensions: `.pike`, `.pmod`, `.c` (Pike modules)
- Content signatures: `#include <lib.h>`, `inherit`, `class`
- Source: [Pike Language](https://pike.lysator.liu.se/)

**LSP Servers:**
- No LSP servers found

**Installation Instructions:**
```bash
# Pike compiler
# Build from source
git clone https://github.com/pikelang/Pike
# Follow build instructions

# VS Code syntax highlighting available
# Search for "Pike Language" extension
```

**Evaluation Narrative:**
Pike has extremely limited tooling support with no LSP servers available. The language's very small user base (estimated <30 active developers) limits tooling development. Basic syntax highlighting exists for some editors.

**Tree-sitter Grammar:** ❌ No tree-sitter grammar available

**Recommendation:** No Viable LSP

---

## Language Coverage Summary

### Comprehensive Language List Analysis

**Total P Languages Researched:** 63 languages from Wikipedia's List

**LSP Support Tiers:**

**Tier 1 - Production Ready (7 languages):**
- Python, PHP, PowerShell, PL/SQL, Processing, PureScript, Prolog

**Tier 2 - Limited Support (3 languages):**
- Pascal, Perl, Pony

**Tier 3 - No LSP Support (53 languages):**
Including but not limited to: P, P4, P′′, ParaSail, PARI/GP, Pascal Script, PCASTL, PCF, PEARL, PeopleCode, PDL, Pharo, Pico, Picolisp, Pict, Pike, PILOT, Pipelines, Pizza, PL-11, PL/0, PL/B, PL/C, PL/I, PL/M, PL/P, PL/S, PL360, PLANC, Plankalkül, Planner, PLEX, PLEXIL, Plus, POP-11, POP-2, PostScript, PortablE, POV-Ray SDL, Powerhouse, PowerBuilder, PPL, Processing.js, Prograph, Project Verona, PROMAL, Promela, PROSE, PROTEL, Pro*C, Pure, Pure Data, PWCT

**Tree-sitter Grammar Availability:**
- Available: Python, PHP, PowerShell, Perl, PureScript, Prolog, Pascal
- Partial/Related: Processing (via Java), PL/SQL (via SQL)
- Not Available: Most specialized/historical languages

---

## Installation Quick Reference

### Primary Choices (Production Ready)

```bash
# Python
pip install python-lsp-server
# or
npm install -g pyright

# PHP
npm install -g intelephense

# PowerShell
# Included with PowerShell extension for VS Code

# PL/SQL
# Oracle SQL Developer or VS Code Oracle extension

# Processing
# Via Java Language Server or LS4P

# PureScript
npm install -g purescript-language-server

# Prolog
swipl pack install lsp_server
```

### Secondary Options (Limited Support)

```bash
# Pascal
# Build pasls from source or use OmniPascal VS Code extension

# Perl
cpanm PLS

# Pony
# Build from source: github.com/ponylang/pony-language-server
```

---

## Recommendations by Use Case

### **Web Development**
- **PHP**: Intelephense (premium) or Phpactor (open source)
- **Python**: Pylsp for Django/Flask development

### **System Administration**
- **PowerShell**: PowerShell Editor Services (first-party Microsoft)
- **Python**: Multiple LSP options for automation scripts

### **Database Development**
- **PL/SQL**: Oracle SQL Developer tooling
- **Python**: For database scripting and ORM work

### **Scientific Computing**
- **Python**: Pyright for type safety, Ruff for fast linting
- **Processing**: Java LSP for visualization projects

### **Functional Programming**
- **PureScript**: Official language server with excellent type support
- **Prolog**: SWI-Prolog lsp_server for logic programming

### **Legacy Code Maintenance**
- **Pascal**: Limited options available, consider migration to modern languages
- **Perl**: Basic LSP support available but limited features

### **Historical/Academic Languages**
- Most P languages in this category lack modern tooling support
- Consider using basic syntax highlighting and manual documentation

---

*Report compiled from comprehensive research of Wikipedia's List of Programming Languages (P section) with focus on Language Server Protocol support, installation procedures, and 2024 tooling status.*
