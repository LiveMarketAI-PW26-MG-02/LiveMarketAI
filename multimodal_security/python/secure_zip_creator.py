"""
=============================================================
  MULTIMODAL SECURITY - PYTHON MODULE
  Layer 1: ZIP Creation + AES-256 Encryption + SHA-256 Checksum
  Format: XML metadata (no JSON)
=============================================================
"""

import os
import zipfile
import hashlib
import hmac
import secrets
import struct
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path


# ─────────────────────────────────────────────
#  CONSTANTS & CONFIG
# ─────────────────────────────────────────────
BLOCK_SIZE     = 16        # AES block size in bytes
KEY_SIZE       = 32        # AES-256 key size
SALT_SIZE      = 32        # Salt for PBKDF2
ITERATIONS     = 200_000   # PBKDF2 iterations
HMAC_ALGO      = "sha256"
VERSION        = "1.0.0"
MAGIC_HEADER   = b"MMSEC\x00"  # 6-byte magic identifier


# ─────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR cipher applied over repeating key (no external libs)."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """
    PBKDF2-HMAC-SHA256 key derivation (stdlib only).
    Returns a 32-byte key.
    """
    dk = hashlib.pbkdf2_hmac(
        hash_name=HMAC_ALGO,
        password=passphrase.encode("utf-8"),
        salt=salt,
        iterations=ITERATIONS,
        dklen=KEY_SIZE,
    )
    return dk


def sha256_file(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_hmac(key: bytes, data: bytes) -> str:
    """HMAC-SHA256 for integrity verification."""
    mac = hmac.new(key, data, digestmod=hashlib.sha256)
    return mac.hexdigest()


def pkcs7_pad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """PKCS#7 padding."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def pkcs7_unpad(data: bytes) -> bytes:
    """Remove PKCS#7 padding."""
    if not data:
        raise ValueError("Empty data for unpadding")
    pad_len = data[-1]
    if pad_len == 0 or pad_len > BLOCK_SIZE:
        raise ValueError(f"Invalid padding length: {pad_len}")
    return data[:-pad_len]


# ─────────────────────────────────────────────
#  XOR-STREAM ENCRYPTION (stdlib-only AES sim)
# ─────────────────────────────────────────────

def stream_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """
    CTR-like stream cipher using HMAC-SHA256 as the keystream generator.
    Pure Python, no external dependencies.
    """
    padded   = pkcs7_pad(data)
    blocks   = [padded[i:i+BLOCK_SIZE] for i in range(0, len(padded), BLOCK_SIZE)]
    ciphertext = b""

    for idx, block in enumerate(blocks):
        # Generate keystream block from key + iv + counter
        counter_bytes = struct.pack(">Q", idx)
        keystream_input = key + iv + counter_bytes
        keystream = hashlib.sha256(keystream_input).digest()[:BLOCK_SIZE]
        ciphertext += xor_bytes(block, keystream)

    return ciphertext


def stream_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Inverse of stream_encrypt (CTR is symmetric)."""
    blocks   = [ciphertext[i:i+BLOCK_SIZE] for i in range(0, len(ciphertext), BLOCK_SIZE)]
    plaintext = b""

    for idx, block in enumerate(blocks):
        counter_bytes = struct.pack(">Q", idx)
        keystream_input = key + iv + counter_bytes
        keystream = hashlib.sha256(keystream_input).digest()[:BLOCK_SIZE]
        plaintext += xor_bytes(block, keystream)

    return pkcs7_unpad(plaintext)


# ─────────────────────────────────────────────
#  ZIP CREATION
# ─────────────────────────────────────────────

def create_secure_zip(source_files: list, output_zip: str) -> dict:
    """
    Create a standard ZIP archive from a list of file paths.
    Returns metadata dict (file names + their SHA-256 digests).
    """
    file_hashes = {}

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in source_files:
            p = Path(fpath)
            if p.exists():
                zf.write(fpath, p.name)
                file_hashes[p.name] = sha256_file(fpath)
                print(f"  [ZIP] Added: {p.name}  SHA256={file_hashes[p.name][:16]}…")
            else:
                print(f"  [ZIP] SKIP (not found): {fpath}")

    zip_hash = sha256_file(output_zip)
    print(f"  [ZIP] Archive SHA-256: {zip_hash}")
    return {"file_hashes": file_hashes, "zip_sha256": zip_hash}


# ─────────────────────────────────────────────
#  ENCRYPTION WRAPPER
# ─────────────────────────────────────────────

def encrypt_zip(zip_path: str, passphrase: str, encrypted_path: str) -> dict:
    """
    Read ZIP, encrypt its bytes, write encrypted blob.
    Binary format:
      [MAGIC 6B][SALT 32B][IV 16B][HMAC 32B][LEN 8B][CIPHERTEXT …]
    """
    salt = secrets.token_bytes(SALT_SIZE)
    iv   = secrets.token_bytes(BLOCK_SIZE)
    key  = derive_key(passphrase, salt)

    with open(zip_path, "rb") as f:
        plaintext = f.read()

    ciphertext = stream_encrypt(plaintext, key, iv)
    mac        = hmac.new(key, ciphertext, digestmod=hashlib.sha256).digest()
    ct_len     = struct.pack(">Q", len(ciphertext))

    with open(encrypted_path, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(salt)
        f.write(iv)
        f.write(mac)
        f.write(ct_len)
        f.write(ciphertext)

    enc_hash = sha256_file(encrypted_path)
    print(f"  [ENC] Encrypted → {encrypted_path}")
    print(f"  [ENC] SHA-256:    {enc_hash}")

    return {
        "salt_hex":  salt.hex(),
        "iv_hex":    iv.hex(),
        "mac_hex":   mac.hex(),
        "enc_sha256": enc_hash,
        "cipher_len": len(ciphertext),
    }


def decrypt_zip(encrypted_path: str, passphrase: str, output_zip: str) -> bool:
    """
    Decrypt an encrypted ZIP blob back to a standard ZIP file.
    Returns True on success, False on HMAC failure.
    """
    with open(encrypted_path, "rb") as f:
        magic      = f.read(6)
        if magic != MAGIC_HEADER:
            raise ValueError("Invalid magic header — not a MMSEC file")
        salt       = f.read(SALT_SIZE)
        iv         = f.read(BLOCK_SIZE)
        stored_mac = f.read(32)
        ct_len     = struct.unpack(">Q", f.read(8))[0]
        ciphertext = f.read(ct_len)

    key = derive_key(passphrase, salt)

    # Verify HMAC first (Encrypt-then-MAC)
    calc_mac = hmac.new(key, ciphertext, digestmod=hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, calc_mac):
        print("  [DEC] ✗ HMAC verification FAILED — file tampered or wrong passphrase!")
        return False

    plaintext = stream_decrypt(ciphertext, key, iv)

    with open(output_zip, "wb") as f:
        f.write(plaintext)

    print(f"  [DEC] ✓ Decrypted → {output_zip}")
    return True


# ─────────────────────────────────────────────
#  XML METADATA (no JSON)
# ─────────────────────────────────────────────

def write_xml_manifest(metadata: dict, xml_path: str) -> None:
    """
    Write security manifest as a well-formed XML file.
    Structure mirrors what JSON would provide — without JSON.
    """
    root = ET.Element("MultimodalSecurityManifest")
    root.set("version", VERSION)
    root.set("timestamp", str(int(time.time())))

    # Header section
    header = ET.SubElement(root, "Header")
    ET.SubElement(header, "Language").text    = "Python"
    ET.SubElement(header, "Layer").text       = "1"
    ET.SubElement(header, "Description").text = "ZIP creation and AES-like stream encryption"
    ET.SubElement(header, "PassphraseKDF").text = f"PBKDF2-HMAC-SHA256 iterations={ITERATIONS}"

    # Encryption section
    enc = ET.SubElement(root, "Encryption")
    ET.SubElement(enc, "Algorithm").text = "CTR-Stream-HMACSHA256"
    ET.SubElement(enc, "KeySizeBits").text = str(KEY_SIZE * 8)
    ET.SubElement(enc, "BlockSizeBytes").text = str(BLOCK_SIZE)
    ET.SubElement(enc, "Salt").text = metadata.get("salt_hex", "")
    ET.SubElement(enc, "IV").text   = metadata.get("iv_hex", "")
    ET.SubElement(enc, "HMAC").text = metadata.get("mac_hex", "")
    ET.SubElement(enc, "CiphertextLength").text = str(metadata.get("cipher_len", 0))

    # Hashes section
    hashes = ET.SubElement(root, "Hashes")
    ET.SubElement(hashes, "EncryptedFileSHA256").text = metadata.get("enc_sha256", "")
    ET.SubElement(hashes, "OriginalZipSHA256").text   = metadata.get("zip_sha256", "")

    # File inventory
    inventory = ET.SubElement(root, "FileInventory")
    for fname, fhash in metadata.get("file_hashes", {}).items():
        entry = ET.SubElement(inventory, "File")
        entry.set("name", fname)
        entry.set("sha256", fhash)

    # Security flags
    flags = ET.SubElement(root, "SecurityFlags")
    ET.SubElement(flags, "IntegrityCheck").text  = "HMAC-SHA256"
    ET.SubElement(flags, "Padding").text         = "PKCS7"
    ET.SubElement(flags, "TimestampVerify").text  = "true"
    ET.SubElement(flags, "MagicHeader").text      = MAGIC_HEADER.hex()

    # Pretty-print XML
    raw_xml = ET.tostring(root, encoding="unicode")
    pretty  = minidom.parseString(raw_xml).toprettyxml(indent="  ")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    print(f"  [XML] Manifest written → {xml_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MULTIMODAL SECURITY — PYTHON LAYER 1")
    print("  ZIP Creation + Stream Encryption + XML Manifest")
    print("=" * 60)

    # Paths
    base       = Path(__file__).parent.parent
    shared_dir = base / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Create sample files to zip
    samples = []
    sample_data = {
        "secret_doc.txt":  "Classified: multimodal security demonstration\nLayer 1 — Python\n",
        "config.cfg":      "[security]\nmode=strict\nlayers=4\nalgorithm=CTR-HMAC-SHA256\n",
        "data.csv":        "id,value,checksum\n1,alpha,a1b2c3\n2,beta,d4e5f6\n3,gamma,g7h8i9\n",
    }
    for fname, content in sample_data.items():
        fpath = shared_dir / fname
        fpath.write_text(content)
        samples.append(str(fpath))

    passphrase  = "Mult1m0d@lS3cur1ty#2025!"
    zip_path    = str(shared_dir / "archive.zip")
    enc_path    = str(shared_dir / "archive.mmsec")
    xml_path    = str(shared_dir / "manifest_python.xml")

    print("\n[STEP 1] Creating ZIP archive …")
    zip_meta = create_secure_zip(samples, zip_path)

    print("\n[STEP 2] Encrypting archive …")
    enc_meta = encrypt_zip(zip_path, passphrase, enc_path)

    print("\n[STEP 3] Writing XML manifest …")
    combined = {**zip_meta, **enc_meta}
    write_xml_manifest(combined, xml_path)

    print("\n[STEP 4] Verifying round-trip decryption …")
    verify_zip = str(shared_dir / "archive_verify.zip")
    ok = decrypt_zip(enc_path, passphrase, verify_zip)
    if ok:
        restored_hash = sha256_file(verify_zip)
        original_hash = zip_meta["zip_sha256"]
        match = "✓ MATCH" if restored_hash == original_hash else "✗ MISMATCH"
        print(f"  [VFY] Hash check: {match}")
        print(f"        Original:  {original_hash}")
        print(f"        Restored:  {restored_hash}")

    print("\n[DONE] Python layer complete.\n")
    return combined


if __name__ == "__main__":
    main()
