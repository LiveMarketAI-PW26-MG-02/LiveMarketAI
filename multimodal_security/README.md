# Multimodal Security 

A **4-language** (Python · R · C++ · Java) security pipeline for creating and
protecting ZIP archives. Every layer adds a distinct cryptographic or analytical
security control. Data is exchanged exclusively via **XML** .

---

## Architecture Overview

```
Source Files
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 · Python                                           │
│  • Creates ZIP archive (DEFLATE compression)                │
│  • Stream-cipher encryption  (CTR + HMAC-SHA256)            │
│  • PBKDF2-HMAC-SHA256 key derivation (200 000 iterations)  │
│  • PKCS#7 padding                                           │
│  • Writes  manifest_python.xml                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ archive.mmsec  +  XML
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 · R                                                │
│  • Shannon entropy analysis        (must be > 7.5 b/B)      │
│  • Chi-squared uniformity test     (p > 0.05)               │
│  • Autocorrelation check           (max|ACF| within 95% CI) │
│  • Wald-Wolfowitz runs test        (p > 0.05)               │
│  • MD5 cross-language checksum                              │
│  • Reads  manifest_python.xml                               │
│  • Writes security_report_r.xml                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ analysis + XML
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 · C++  (C++17, pure stdlib)                        │
│  • Pure-C++ SHA-256 implementation                          │
│  • Feistel cipher (64-bit block, 16 rounds, CBC mode)       │
│  • PKCS#7 padding                                           │
│  • CRC-32 integrity tag (prepended to blob)                 │
│  • Binary blob format: MAGIC + CRC + ciphertext             │
│  • Writes manifest_cpp.xml                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ .mmsec blob + XML
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 · Java  (standard library only)                    │
│  • AES-256-GCM authenticated encryption                     │
│  • PBKDF2-HMAC-SHA256 (310 000 iterations, OWASP 2024)     │
│  • GCM 128-bit auth tag (tamper-evident, no padding needed) │
│  • HMAC-SHA256 blob signature                               │
│  • Binary blob: MAGIC+SALT+IV+TAG+LEN+CIPHERTEXT            │
│  • Writes manifest_java.xml (DOM + XSLT pretty-print)       │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
multimodal_security/
├── orchestrate.py              ← Master runner (runs all 4 layers)
│
├── python/
│   └── secure_zip_creator.py  ← Layer 1 — ZIP + stream encryption
│
├── r/
│   └── integrity_verifier.R   ← Layer 2 — statistical analysis
│
├── cpp/
│   └── secure_layer.cpp       ← Layer 3 — Feistel cipher + CRC-32
│
├── java/
│   └── SecureZipManager.java  ← Layer 4 — AES-256-GCM + PBKDF2
│
└── shared/                    ← Runtime artifacts (auto-created)
    ├── archive.zip             ← Original ZIP
    ├── archive.mmsec           ← Python-encrypted blob
    ├── archive_cpp.mmsec       ← C++-encrypted blob
    ├── archive_java.mmsec      ← Java-encrypted blob (AES-GCM)
    ├── manifest_python.xml     ← Python encryption metadata
    ├── manifest_cpp.xml        ← C++ encryption metadata
    ├── manifest_java.xml       ← Java encryption metadata
    └── security_report_r.xml  ← R statistical analysis report
```

---

## Quick Start

### Run everything at once
```bash
cd multimodal_security
python3 orchestrate.py
```

### Run individual layers

**Layer 1 — Python**
```bash
python3 python/secure_zip_creator.py
```

**Layer 2 — R**
```bash
Rscript r/integrity_verifier.R
```

**Layer 3 — C++**
```bash
g++ -std=c++17 -O2 -o cpp/mmsec_cpp cpp/secure_layer.cpp

./cpp/mmsec_cpp encrypt  shared/archive.zip shared/archive_cpp.mmsec "MyKey"
./cpp/mmsec_cpp decrypt  shared/archive_cpp.mmsec shared/out.zip     "MyKey"
./cpp/mmsec_cpp analyse  shared/archive_cpp.mmsec
./cpp/mmsec_cpp manifest shared/archive_cpp.mmsec shared/manifest_cpp.xml
```

**Layer 4 — Java**
```bash
javac java/SecureZipManager.java -d java/
java -cp java/ SecureZipManager shared/
```

---

## Security Controls Per Layer

| Control                    | Python | R  | C++ | Java |
|----------------------------|--------|----|-----|------|
| Stream cipher (CTR)        | ✓      |    |     |      |
| Block cipher (Feistel-CBC) |        |    | ✓   |      |
| AES-256-GCM                |        |    |     | ✓    |
| PBKDF2 key derivation      | ✓      |    |     | ✓    |
| PKCS#7 padding             | ✓      |    | ✓   |      |
| HMAC-SHA256 signing        | ✓      |    |     | ✓    |
| SHA-256 checksum           | ✓      | ✓  | ✓   | ✓    |
| CRC-32 checksum            |        |    | ✓   |      |
| Shannon entropy test       |        | ✓  |     |      |
| Chi-squared test           |        | ✓  |     |      |
| Autocorrelation test       |        | ✓  |     |      |
| Runs (randomness) test     |        | ✓  |     |      |
| Magic header validation    | ✓      |    | ✓   | ✓    |
| XML-only metadata          | ✓      | ✓  | ✓   | ✓    |

---

## Encryption Formats

### Python blob (`archive.mmsec`)
```
[MAGIC "MMSEC\x00" — 6B]
[SALT  — 32B]
[IV    — 16B]
[HMAC  — 32B]
[LEN   — 8B big-endian]
[CIPHERTEXT — variable]
```

### C++ blob (`archive_cpp.mmsec`)
```
[MAGIC "MMC+" — 4B]
[VERSION — 1B]
[CRC32_PLAINTEXT — 4B]
[IV — 8B (prepended by encrypt)]
[CIPHERTEXT-FEISTEL-CBC — variable]
```

### Java blob (`archive_java.mmsec`)
```
[MAGIC "MMSJ" — 4B]
[SALT — 32B]
[IV/NONCE — 12B]
[GCM-AUTH-TAG — 16B]
[ORIGINAL_SIZE — 8B big-endian]
[AES-256-GCM-CIPHERTEXT — variable]
```

---

## Requirements

| Layer  | Requirement              |
|--------|--------------------------|
| Python | Python 3.8+, stdlib only |
| R      | R 4.0+, `tools` package  |
| C++    | g++ / clang++ (C++17)    |
| Java   | JDK 11+                  |

No third-party libraries required. No JSON used anywhere.
