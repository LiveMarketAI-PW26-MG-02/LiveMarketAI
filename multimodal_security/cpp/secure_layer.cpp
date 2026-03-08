/*
 * =============================================================
 *  MULTIMODAL SECURITY — C++ MODULE
 *  Layer 3: Binary Encryption Layer (Feistel + Checksum)
 *           XML Manifest Reader/Writer (no JSON)
 * =============================================================
 *
 *  Build:
 *    g++ -std=c++17 -O2 -o mmsec_cpp secure_layer.cpp
 *
 *  Usage:
 *    ./mmsec_cpp encrypt  <input_file> <output_file> <key>
 *    ./mmsec_cpp decrypt  <input_file> <output_file> <key>
 *    ./mmsec_cpp analyse  <file>
 *    ./mmsec_cpp manifest <input_file> <xml_out>
 * =============================================================
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────
static constexpr uint8_t  MAGIC[4]    = {0x4D, 0x4D, 0x43, 0x2B}; // "MMC+"
static constexpr uint8_t  VERSION_BYTE = 0x03;
static constexpr size_t   KEY_BYTES   = 32;
static constexpr size_t   BLOCK_SIZE  = 8;    // 64-bit Feistel block
static constexpr int      ROUNDS      = 16;


// ─────────────────────────────────────────────
//  SHA-256 IMPLEMENTATION (pure C++, no openssl)
// ─────────────────────────────────────────────
static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
    0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
    0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
    0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
    0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
    0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

class SHA256 {
    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> buf_{};
    uint64_t bitcount_ = 0;
    size_t   buflen_   = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    static uint32_t ch(uint32_t e, uint32_t f, uint32_t g)  { return (e & f) ^ (~e & g); }
    static uint32_t maj(uint32_t a, uint32_t b, uint32_t c) { return (a & b) ^ (a & c) ^ (b & c); }
    static uint32_t ep0(uint32_t a) { return rotr(a,2) ^ rotr(a,13) ^ rotr(a,22); }
    static uint32_t ep1(uint32_t e) { return rotr(e,6) ^ rotr(e,11) ^ rotr(e,25); }
    static uint32_t sig0(uint32_t x){ return rotr(x,7) ^ rotr(x,18) ^ (x>>3); }
    static uint32_t sig1(uint32_t x){ return rotr(x,17) ^ rotr(x,19) ^ (x>>10); }

    void compress(const uint8_t* blk) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t(blk[i*4])   << 24) | (uint32_t(blk[i*4+1]) << 16)
                 | (uint32_t(blk[i*4+2]) <<  8) |  uint32_t(blk[i*4+3]);
        }
        for (int i = 16; i < 64; i++)
            w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];

        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + ep1(e) + ch(e,f,g) + SHA256_K[i] + w[i];
            uint32_t t2 = ep0(a) + maj(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

public:
    SHA256() { reset(); }
    void reset() {
        state_ = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        bitcount_ = 0; buflen_ = 0;
    }
    void update(const uint8_t* data, size_t len) {
        bitcount_ += uint64_t(len) * 8;
        while (len > 0) {
            size_t space = 64 - buflen_;
            size_t copy  = std::min(space, len);
            std::memcpy(buf_.data() + buflen_, data, copy);
            buflen_ += copy; data += copy; len -= copy;
            if (buflen_ == 64) { compress(buf_.data()); buflen_ = 0; }
        }
    }
    std::array<uint8_t, 32> digest() {
        uint8_t tmp = 0x80;
        update(&tmp, 1);
        while (buflen_ != 56) { tmp = 0; update(&tmp, 1); }
        for (int i = 7; i >= 0; i--) { tmp = uint8_t(bitcount_ >> (i*8)); update(&tmp, 1); }
        std::array<uint8_t, 32> out{};
        for (int i = 0; i < 8; i++) {
            out[i*4+0] = state_[i] >> 24; out[i*4+1] = (state_[i] >> 16) & 0xFF;
            out[i*4+2] = (state_[i] >> 8) & 0xFF; out[i*4+3] = state_[i] & 0xFF;
        }
        return out;
    }
    static std::string hex(const std::array<uint8_t,32>& d) {
        std::ostringstream oss;
        for (auto b : d) oss << std::hex << std::setw(2) << std::setfill('0') << int(b);
        return oss.str();
    }
};


// ─────────────────────────────────────────────
//  KEY SCHEDULE (derive 16 round sub-keys)
// ─────────────────────────────────────────────
static std::array<uint32_t, ROUNDS> derive_subkeys(const std::string& passphrase) {
    SHA256 h;
    h.update(reinterpret_cast<const uint8_t*>(passphrase.data()), passphrase.size());
    auto digest = h.digest();

    std::array<uint32_t, ROUNDS> skeys{};
    for (int r = 0; r < ROUNDS; r++) {
        int base = (r * 2) % 32;
        skeys[r] = (uint32_t(digest[base])   << 24)
                 | (uint32_t(digest[(base+1)%32]) << 16)
                 | (uint32_t(digest[(base+2)%32]) <<  8)
                 |  uint32_t(digest[(base+3)%32]);
        // mix with round constant
        skeys[r] ^= 0xA5A5A5A5u ^ uint32_t(r * 0x9E3779B9u);
    }
    return skeys;
}


// ─────────────────────────────────────────────
//  FEISTEL CIPHER (64-bit block, 16 rounds)
// ─────────────────────────────────────────────
static uint32_t F(uint32_t x, uint32_t k) {
    // Non-linear mixing function
    x ^= k;
    x += (x << 6);
    x ^= (x >> 11);
    x += (x << 22);
    x ^= k;
    return x;
}

static uint64_t feistel_encrypt(uint64_t block, const std::array<uint32_t,ROUNDS>& skeys) {
    uint32_t L = uint32_t(block >> 32);
    uint32_t R = uint32_t(block & 0xFFFFFFFF);
    for (int r = 0; r < ROUNDS; r++) {
        uint32_t tmp = R ^ F(L, skeys[r]);
        R = L;
        L = tmp;
    }
    return (uint64_t(L) << 32) | R;
}

static uint64_t feistel_decrypt(uint64_t block, const std::array<uint32_t,ROUNDS>& skeys) {
    uint32_t L = uint32_t(block >> 32);
    uint32_t R = uint32_t(block & 0xFFFFFFFF);
    // Each encrypt round: L_new = R ^ F(L, k), R_new = L
    // Inverse:            L_old = R_new, R_old = L_new ^ F(R_new, k)
    for (int r = ROUNDS - 1; r >= 0; r--) {
        uint32_t tmp = R;           // R_new = L_old
        R = L ^ F(R, skeys[r]);     // R_old = L_new ^ F(R_new, k)
        L = tmp;                    // L_old
    }
    return (uint64_t(L) << 32) | R;
}


// ─────────────────────────────────────────────
//  CBC-LIKE ENCRYPT / DECRYPT
// ─────────────────────────────────────────────
static std::vector<uint8_t> encrypt_data(
    const std::vector<uint8_t>& plaintext,
    const std::string& passphrase)
{
    auto skeys = derive_subkeys(passphrase);

    // Pad to block boundary
    std::vector<uint8_t> padded = plaintext;
    size_t pad = BLOCK_SIZE - (padded.size() % BLOCK_SIZE);
    padded.insert(padded.end(), pad, uint8_t(pad));

    // IV = time-seeded
    uint64_t iv = uint64_t(std::time(nullptr)) ^ 0xDEADBEEFCAFEBABEull;

    std::vector<uint8_t> cipher;
    // Prepend IV
    for (int i = 7; i >= 0; i--) cipher.push_back(uint8_t(iv >> (i*8)));

    uint64_t prev = iv;
    for (size_t i = 0; i < padded.size(); i += BLOCK_SIZE) {
        uint64_t block = 0;
        for (int j = 0; j < 8; j++)
            block = (block << 8) | padded[i + j];
        block ^= prev;                          // CBC XOR
        block = feistel_encrypt(block, skeys);
        prev  = block;
        for (int j = 7; j >= 0; j--) cipher.push_back(uint8_t(block >> (j*8)));
    }

    return cipher;
}

static std::vector<uint8_t> decrypt_data(
    const std::vector<uint8_t>& ciphertext,
    const std::string& passphrase)
{
    if (ciphertext.size() < BLOCK_SIZE + BLOCK_SIZE)
        throw std::runtime_error("Ciphertext too short");

    auto skeys = derive_subkeys(passphrase);

    // Extract IV
    uint64_t prev = 0;
    for (int i = 0; i < 8; i++) prev = (prev << 8) | ciphertext[i];

    std::vector<uint8_t> plain;
    for (size_t i = BLOCK_SIZE; i < ciphertext.size(); i += BLOCK_SIZE) {
        uint64_t block = 0;
        for (int j = 0; j < 8; j++) block = (block << 8) | ciphertext[i + j];
        uint64_t dec = feistel_decrypt(block, skeys) ^ prev;
        prev = block;
        for (int j = 7; j >= 0; j--) plain.push_back(uint8_t(dec >> (j*8)));
    }

    // Remove padding
    if (!plain.empty()) {
        uint8_t pad = plain.back();
        if (pad > 0 && pad <= uint8_t(BLOCK_SIZE))
            plain.resize(plain.size() - pad);
    }

    return plain;
}


// ─────────────────────────────────────────────
//  CRC-32 CHECKSUM
// ─────────────────────────────────────────────
static uint32_t crc32(const std::vector<uint8_t>& data) {
    static const auto table = [](){
        std::array<uint32_t,256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    uint32_t crc = 0xFFFFFFFFu;
    for (auto b : data) crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}


// ─────────────────────────────────────────────
//  BINARY ENTROPY
// ─────────────────────────────────────────────
static double entropy(const std::vector<uint8_t>& data) {
    if (data.empty()) return 0.0;
    std::array<size_t,256> freq{};
    for (auto b : data) freq[b]++;
    double H = 0.0;
    double n = double(data.size());
    for (auto f : freq) {
        if (f > 0) {
            double p = f / n;
            H -= p * std::log2(p);
        }
    }
    return H;
}


// ─────────────────────────────────────────────
//  FILE I/O HELPERS
// ─────────────────────────────────────────────
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    return {std::istreambuf_iterator<char>(f), {}};
}

static void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

static std::string sha256_file(const std::string& path) {
    auto data = read_file(path);
    SHA256 h;
    h.update(data.data(), data.size());
    return SHA256::hex(h.digest());
}

static std::string hex_str(uint32_t v) {
    std::ostringstream oss;
    oss << std::hex << std::setw(8) << std::setfill('0') << v;
    return oss.str();
}


// ─────────────────────────────────────────────
//  XML MANIFEST WRITER
// ─────────────────────────────────────────────
static void write_xml_manifest(
    const std::string& input_path,
    const std::string& xml_path,
    const std::string& sha256_val,
    uint32_t           crc_val,
    double             ent,
    size_t             file_size)
{
    auto now   = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
                     now.time_since_epoch()).count();

    std::ofstream xml(xml_path);
    if (!xml) throw std::runtime_error("Cannot write XML: " + xml_path);

    xml << R"(<?xml version="1.0" encoding="UTF-8"?>)" << "\n"
        << R"(<MultimodalSecurityManifest language="C++" layer="3">)" << "\n"
        << "  <Timestamp>" << epoch << "</Timestamp>\n"
        << "  <Description>Feistel-cipher encryption with CRC-32 and SHA-256 verification</Description>\n"
        << "\n"
        << "  <Encryption>\n"
        << "    <Algorithm>Feistel-CBC-16rounds</Algorithm>\n"
        << "    <BlockSizeBits>64</BlockSizeBits>\n"
        << "    <Rounds>16</Rounds>\n"
        << "    <KeyDerivation>SHA256-based schedule</KeyDerivation>\n"
        << "    <Padding>PKCS7</Padding>\n"
        << "  </Encryption>\n"
        << "\n"
        << "  <FileInfo>\n"
        << "    <Filename>" << fs::path(input_path).filename().string() << "</Filename>\n"
        << "    <SizeBytes>" << file_size << "</SizeBytes>\n"
        << "    <SHA256>" << sha256_val << "</SHA256>\n"
        << "    <CRC32>" << hex_str(crc_val) << "</CRC32>\n"
        << "    <Entropy>" << std::fixed << std::setprecision(6) << ent << "</Entropy>\n"
        << "  </FileInfo>\n"
        << "\n"
        << "  <SecurityChecks>\n"
        << "    <MagicHeader>4D4D432B</MagicHeader>\n"
        << "    <VersionByte>03</VersionByte>\n"
        << "    <IntegrityAlgo>CRC32 + SHA256</IntegrityAlgo>\n"
        << "    <CipherMode>CBC</CipherMode>\n"
        << "  </SecurityChecks>\n"
        << "</MultimodalSecurityManifest>\n";

    xml.close();
    std::cout << "  [XML] Manifest written → " << fs::path(xml_path).filename() << "\n";
}


// ─────────────────────────────────────────────
//  ENCRYPT COMMAND
// ─────────────────────────────────────────────
static void cmd_encrypt(const std::string& in, const std::string& out, const std::string& key) {
    std::cout << "\n[ENCRYPT] " << in << " → " << out << "\n";
    auto plain  = read_file(in);
    auto cipher = encrypt_data(plain, key);

    uint32_t crc_plain  = crc32(plain);
    uint32_t crc_cipher = crc32(cipher);

    // Write header: MAGIC(4) + VERSION(1) + CRC_PLAIN(4) + data
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), MAGIC, MAGIC + 4);
    blob.push_back(VERSION_BYTE);
    blob.push_back(uint8_t(crc_plain >> 24));
    blob.push_back(uint8_t(crc_plain >> 16));
    blob.push_back(uint8_t(crc_plain >>  8));
    blob.push_back(uint8_t(crc_plain));
    blob.insert(blob.end(), cipher.begin(), cipher.end());
    write_file(out, blob);

    SHA256 h;
    h.update(blob.data(), blob.size());
    auto dig = h.digest();

    std::cout << "  Input size   : " << plain.size()  << " bytes\n";
    std::cout << "  Output size  : " << blob.size()   << " bytes\n";
    std::cout << "  CRC32 plain  : " << hex_str(crc_plain)  << "\n";
    std::cout << "  CRC32 cipher : " << hex_str(crc_cipher) << "\n";
    std::cout << "  SHA256       : " << SHA256::hex(dig) << "\n";
}


// ─────────────────────────────────────────────
//  DECRYPT COMMAND
// ─────────────────────────────────────────────
static void cmd_decrypt(const std::string& in, const std::string& out, const std::string& key) {
    std::cout << "\n[DECRYPT] " << in << " → " << out << "\n";
    auto blob = read_file(in);

    // Verify magic
    if (blob.size() < 9 || std::memcmp(blob.data(), MAGIC, 4) != 0)
        throw std::runtime_error("Invalid magic header");
    if (blob[4] != VERSION_BYTE)
        throw std::runtime_error("Unsupported version byte");

    uint32_t stored_crc = (uint32_t(blob[5]) << 24) | (uint32_t(blob[6]) << 16)
                        | (uint32_t(blob[7]) <<  8) |  uint32_t(blob[8]);

    std::vector<uint8_t> cipher(blob.begin() + 9, blob.end());
    auto plain = decrypt_data(cipher, key);

    uint32_t calc_crc = crc32(plain);
    bool ok = (stored_crc == calc_crc);

    std::cout << "  Stored CRC32 : " << hex_str(stored_crc) << "\n";
    std::cout << "  Actual CRC32 : " << hex_str(calc_crc)   << "\n";
    std::cout << "  Integrity    : " << (ok ? "✓ PASS" : "✗ FAIL") << "\n";

    if (!ok) throw std::runtime_error("CRC32 mismatch — data tampered or wrong key!");
    write_file(out, plain);
    std::cout << "  Output size  : " << plain.size() << " bytes\n";
}


// ─────────────────────────────────────────────
//  ANALYSE COMMAND
// ─────────────────────────────────────────────
static void cmd_analyse(const std::string& in) {
    std::cout << "\n[ANALYSE] " << in << "\n";
    auto data = read_file(in);
    double ent = entropy(data);
    uint32_t crc = crc32(data);
    std::cout << "  Size    : " << data.size() << " bytes\n";
    std::cout << "  Entropy : " << std::fixed << std::setprecision(4) << ent << " bits/byte\n";
    std::cout << "  CRC32   : " << hex_str(crc) << "\n";
    std::cout << "  SHA256  : " << sha256_file(in) << "\n";
    std::cout << "  Quality : " << (ent > 7.5 ? "HIGH ✓" : "LOW ✗") << "\n";
}


// ─────────────────────────────────────────────
//  MANIFEST COMMAND
// ─────────────────────────────────────────────
static void cmd_manifest(const std::string& in, const std::string& xml_out) {
    auto data  = read_file(in);
    double ent = entropy(data);
    uint32_t crc = crc32(data);
    SHA256 h;
    h.update(data.data(), data.size());
    auto dig = h.digest();
    write_xml_manifest(in, xml_out, SHA256::hex(dig), crc, ent, data.size());
}


// ─────────────────────────────────────────────
//  SELF-TEST
// ─────────────────────────────────────────────
static void self_test() {
    std::cout << "\n[SELF-TEST] Running Feistel cipher round-trip …\n";
    std::string key = "TestKey#2025!MultimodalSecurity";
    std::vector<uint8_t> msg(64);
    for (size_t i = 0; i < msg.size(); i++) msg[i] = uint8_t(i * 7 + 13);

    auto cipher = encrypt_data(msg, key);
    auto plain2 = decrypt_data(cipher, key);

    bool ok = (plain2 == msg);
    std::cout << "  Feistel round-trip: " << (ok ? "✓ PASS" : "✗ FAIL") << "\n";

    // CRC-32 sanity
    std::vector<uint8_t> zeros(256, 0);
    uint32_t crc_z = crc32(zeros);
    std::cout << "  CRC-32 sanity:      " << (crc_z != 0 ? "✓ PASS" : "✗ FAIL") << "\n";
}


// ─────────────────────────────────────────────
//  MAIN (standalone demo if no args)
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "============================================================\n"
              << "  MULTIMODAL SECURITY — C++ LAYER 3\n"
              << "  Feistel Cipher + CRC-32 + SHA-256 + XML Manifest\n"
              << "============================================================\n";

    // Self-test always runs
    self_test();

    if (argc < 2) {
        std::cout << "\nUsage:\n"
                  << "  mmsec_cpp encrypt  <in> <out> <key>\n"
                  << "  mmsec_cpp decrypt  <in> <out> <key>\n"
                  << "  mmsec_cpp analyse  <file>\n"
                  << "  mmsec_cpp manifest <file> <xml_out>\n";
        return 0;
    }

    try {
        std::string cmd = argv[1];
        if (cmd == "encrypt" && argc == 5)
            cmd_encrypt(argv[2], argv[3], argv[4]);
        else if (cmd == "decrypt" && argc == 5)
            cmd_decrypt(argv[2], argv[3], argv[4]);
        else if (cmd == "analyse" && argc == 3)
            cmd_analyse(argv[2]);
        else if (cmd == "manifest" && argc == 4)
            cmd_manifest(argv[2], argv[3]);
        else {
            std::cerr << "Unknown command or wrong arg count.\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n[DONE] C++ layer complete.\n";
    return 0;
}
