/*
 * =============================================================
 *  MULTIMODAL SECURITY — JAVA MODULE
 *  Layer 4: Final ZIP Packaging + Digital Signature + XML Manifest
 *  No JSON — uses XML and binary formats throughout
 * =============================================================
 *
 *  Compile : javac SecureZipManager.java
 *  Run     : java SecureZipManager [shared_dir]
 * =============================================================
 */

import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.spec.*;
import java.util.*;
import java.util.zip.*;
import java.util.stream.*;
import java.time.*;
import java.time.format.*;
import javax.crypto.*;
import javax.crypto.spec.*;
import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import org.w3c.dom.*;


public class SecureZipManager {

    // ─────────────────────────────────────────────
    //  CONSTANTS
    // ─────────────────────────────────────────────
    private static final String VERSION      = "1.0.0";
    private static final String LAYER        = "4";
    private static final String MAGIC        = "MMSJ";           // Java magic
    private static final int    PBKDF2_ITER  = 310_000;          // OWASP 2024 recommendation
    private static final int    KEY_LEN_BITS = 256;
    private static final int    SALT_BYTES   = 32;
    private static final int    IV_BYTES     = 12;               // GCM nonce
    private static final int    GCM_TAG_BITS = 128;
    private static final String KDF_ALGO     = "PBKDF2WithHmacSHA256";
    private static final String CIPHER_ALGO  = "AES/GCM/NoPadding";
    private static final String KEY_ALGO     = "AES";

    // ─────────────────────────────────────────────
    //  INNER: ENCRYPTION RESULT
    // ─────────────────────────────────────────────
    static class EncResult {
        byte[] salt, iv, ciphertext, tag;
        String sha256;
        long   originalSize;
    }


    // ─────────────────────────────────────────────
    //  SHA-256 UTILITY
    // ─────────────────────────────────────────────
    static String sha256hex(byte[] data) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] d = md.digest(data);
        StringBuilder sb = new StringBuilder();
        for (byte b : d) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    static String sha256file(Path p) throws Exception {
        return sha256hex(Files.readAllBytes(p));
    }


    // ─────────────────────────────────────────────
    //  KEY DERIVATION (PBKDF2-HMAC-SHA256)
    // ─────────────────────────────────────────────
    static SecretKey deriveKey(char[] passphrase, byte[] salt) throws Exception {
        PBEKeySpec spec = new PBEKeySpec(passphrase, salt, PBKDF2_ITER, KEY_LEN_BITS);
        SecretKeyFactory skf = SecretKeyFactory.getInstance(KDF_ALGO);
        byte[] keyBytes = skf.generateSecret(spec).getEncoded();
        spec.clearPassword();
        return new SecretKeySpec(keyBytes, KEY_ALGO);
    }


    // ─────────────────────────────────────────────
    //  AES-256-GCM ENCRYPT
    // ─────────────────────────────────────────────
    static EncResult encryptAESGCM(byte[] plaintext, char[] passphrase) throws Exception {
        SecureRandom rng  = new SecureRandom();
        byte[] salt = new byte[SALT_BYTES];
        byte[] iv   = new byte[IV_BYTES];
        rng.nextBytes(salt);
        rng.nextBytes(iv);

        SecretKey key = deriveKey(passphrase, salt);
        Cipher cipher = Cipher.getInstance(CIPHER_ALGO);
        GCMParameterSpec gcmSpec = new GCMParameterSpec(GCM_TAG_BITS, iv);
        cipher.init(Cipher.ENCRYPT_MODE, key, gcmSpec);
        byte[] ct = cipher.doFinal(plaintext);   // last GCM_TAG_BITS/8 bytes = auth tag

        EncResult r = new EncResult();
        r.salt         = salt;
        r.iv           = iv;
        r.ciphertext   = Arrays.copyOf(ct, ct.length - 16);
        r.tag          = Arrays.copyOfRange(ct, ct.length - 16, ct.length);
        r.sha256       = sha256hex(ct);
        r.originalSize = plaintext.length;
        return r;
    }


    // ─────────────────────────────────────────────
    //  AES-256-GCM DECRYPT
    // ─────────────────────────────────────────────
    static byte[] decryptAESGCM(EncResult r, char[] passphrase) throws Exception {
        SecretKey key = deriveKey(passphrase, r.salt);
        Cipher cipher = Cipher.getInstance(CIPHER_ALGO);
        GCMParameterSpec gcmSpec = new GCMParameterSpec(GCM_TAG_BITS, r.iv);
        cipher.init(Cipher.DECRYPT_MODE, key, gcmSpec);

        // Reassemble ciphertext + tag
        byte[] combined = new byte[r.ciphertext.length + r.tag.length];
        System.arraycopy(r.ciphertext, 0, combined, 0, r.ciphertext.length);
        System.arraycopy(r.tag, 0, combined, r.ciphertext.length, r.tag.length);

        return cipher.doFinal(combined);   // throws AEADBadTagException if tampered
    }


    // ─────────────────────────────────────────────
    //  HMAC-SHA256
    // ─────────────────────────────────────────────
    static String hmacSHA256(byte[] key, byte[] data) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key, "HmacSHA256"));
        byte[] sig = mac.doFinal(data);
        StringBuilder sb = new StringBuilder();
        for (byte b : sig) sb.append(String.format("%02x", b));
        return sb.toString();
    }


    // ─────────────────────────────────────────────
    //  CREATE SECURE ZIP (Layer 4 final packaging)
    // ─────────────────────────────────────────────
    static Map<String, String> createFinalZip(
            List<Path> files, Path outputZip) throws Exception {

        Map<String, String> hashes = new LinkedHashMap<>();
        try (ZipOutputStream zos = new ZipOutputStream(
                new BufferedOutputStream(Files.newOutputStream(outputZip)))) {

            zos.setLevel(Deflater.BEST_COMPRESSION);
            zos.setMethod(ZipOutputStream.DEFLATED);

            for (Path p : files) {
                if (!Files.exists(p)) continue;
                byte[] data = Files.readAllBytes(p);
                String hash = sha256hex(data);
                hashes.put(p.getFileName().toString(), hash);

                ZipEntry entry = new ZipEntry(p.getFileName().toString());
                entry.setSize(data.length);
                entry.setComment("SHA256=" + hash);
                zos.putNextEntry(entry);
                zos.write(data);
                zos.closeEntry();
                System.out.printf("  [ZIP] %-30s SHA256=%s…%n",
                    p.getFileName(), hash.substring(0, 16));
            }
        }
        return hashes;
    }


    // ─────────────────────────────────────────────
    //  WRITE BINARY SECURE BLOB
    //  Format: [MAGIC 4B][SALT 32B][IV 12B][TAG 16B][ORIG_LEN 8B][CIPHERTEXT…]
    // ─────────────────────────────────────────────
    static void writeBinaryBlob(Path output, EncResult r) throws Exception {
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(output)))) {
            dos.write(MAGIC.getBytes(StandardCharsets.US_ASCII));  // 4 bytes
            dos.write(r.salt);       // 32 bytes
            dos.write(r.iv);         // 12 bytes
            dos.write(r.tag);        // 16 bytes
            dos.writeLong(r.originalSize);  // 8 bytes
            dos.write(r.ciphertext);
        }
    }

    static EncResult readBinaryBlob(Path input) throws Exception {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(input)))) {
            byte[] magic = dis.readNBytes(4);
            if (!new String(magic, StandardCharsets.US_ASCII).equals(MAGIC))
                throw new SecurityException("Invalid magic header in blob");

            EncResult r = new EncResult();
            r.salt         = dis.readNBytes(SALT_BYTES);
            r.iv           = dis.readNBytes(IV_BYTES);
            r.tag          = dis.readNBytes(16);
            r.originalSize = dis.readLong();
            r.ciphertext   = dis.readAllBytes();
            return r;
        }
    }


    // ─────────────────────────────────────────────
    //  XML MANIFEST WRITER (no JSON)
    // ─────────────────────────────────────────────
    static void writeXMLManifest(
            Path xmlPath,
            Map<String, String> fileHashes,
            EncResult enc,
            String zipSHA256,
            String blobSHA256,
            String hmacSig) throws Exception {

        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        DocumentBuilder db = dbf.newDocumentBuilder();
        Document doc = db.newDocument();

        // Root
        Element root = doc.createElement("MultimodalSecurityManifest");
        root.setAttribute("language", "Java");
        root.setAttribute("layer", LAYER);
        root.setAttribute("version", VERSION);
        root.setAttribute("timestamp",
            DateTimeFormatter.ISO_INSTANT.format(Instant.now()));
        doc.appendChild(root);

        // Header
        Element header = doc.createElement("Header");
        addText(doc, header, "Description",
            "AES-256-GCM encrypted ZIP with PBKDF2 key derivation and HMAC-SHA256 signing");
        addText(doc, header, "KDFAlgorithm", KDF_ALGO);
        addText(doc, header, "KDFIterations", String.valueOf(PBKDF2_ITER));
        addText(doc, header, "CipherAlgorithm", CIPHER_ALGO);
        addText(doc, header, "TagLengthBits", String.valueOf(GCM_TAG_BITS));
        root.appendChild(header);

        // Encryption params
        Element encEl = doc.createElement("EncryptionParams");
        addText(doc, encEl, "Salt",    bytesToHex(enc.salt));
        addText(doc, encEl, "IV",      bytesToHex(enc.iv));
        addText(doc, encEl, "AuthTag", bytesToHex(enc.tag));
        addText(doc, encEl, "OriginalSizeBytes", String.valueOf(enc.originalSize));
        root.appendChild(encEl);

        // Hashes
        Element hashes = doc.createElement("Hashes");
        addText(doc, hashes, "ZipSHA256",  zipSHA256);
        addText(doc, hashes, "BlobSHA256", blobSHA256);
        addText(doc, hashes, "HMACSHA256", hmacSig);
        root.appendChild(hashes);

        // File inventory
        Element inventory = doc.createElement("FileInventory");
        for (Map.Entry<String, String> e : fileHashes.entrySet()) {
            Element file = doc.createElement("File");
            file.setAttribute("name", e.getKey());
            file.setAttribute("sha256", e.getValue());
            inventory.appendChild(file);
        }
        root.appendChild(inventory);

        // Security profile
        Element profile = doc.createElement("SecurityProfile");
        addText(doc, profile, "ConfidentialityLayer", "AES-256-GCM");
        addText(doc, profile, "IntegrityLayer",       "GCM-AuthTag + HMAC-SHA256");
        addText(doc, profile, "AuthenticationLayer",  "PBKDF2-HMAC-SHA256");
        addText(doc, profile, "MagicHeader",          MAGIC);
        addText(doc, profile, "Language",             "Java");
        addText(doc, profile, "NoJSONUsed",           "true");
        root.appendChild(profile);

        // Serialise with pretty-print
        Transformer tf = TransformerFactory.newInstance().newTransformer();
        tf.setOutputProperty(OutputKeys.INDENT, "yes");
        tf.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        tf.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
        tf.setOutputProperty(OutputKeys.STANDALONE, "yes");
        tf.transform(new DOMSource(doc), new StreamResult(xmlPath.toFile()));

        System.out.println("  [XML] Manifest written → " + xmlPath.getFileName());
    }

    private static void addText(Document doc, Element parent, String tag, String text) {
        Element el = doc.createElement(tag);
        el.setTextContent(text);
        parent.appendChild(el);
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) sb.append(String.format("%02x", b));
        return sb.toString();
    }


    // ─────────────────────────────────────────────
    //  VERIFY ROUND-TRIP
    // ─────────────────────────────────────────────
    static boolean verifyRoundTrip(
            Path blobPath, Path verifyZip,
            char[] passphrase, String expectedSHA256) throws Exception {

        EncResult r = readBinaryBlob(blobPath);
        byte[] plain;
        try {
            plain = decryptAESGCM(r, passphrase);
        } catch (AEADBadTagException e) {
            System.out.println("  [VFY] ✗ GCM authentication tag FAILED — tampered or wrong key!");
            return false;
        }
        Files.write(verifyZip, plain);
        String actualSHA = sha256hex(plain);
        boolean match = actualSHA.equals(expectedSHA256);
        System.out.printf("  [VFY] Expected SHA256: %s%n", expectedSHA256);
        System.out.printf("  [VFY] Actual   SHA256: %s%n", actualSHA);
        System.out.println("  [VFY] Match: " + (match ? "✓ YES" : "✗ NO"));
        return match;
    }


    // ─────────────────────────────────────────────
    //  MAIN
    // ─────────────────────────────────────────────
    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("  MULTIMODAL SECURITY — JAVA LAYER 4");
        System.out.println("  AES-256-GCM + PBKDF2 + HMAC-SHA256 + XML Manifest");
        System.out.println("============================================================");

        // Resolve shared directory
        Path sharedDir;
        if (args.length > 0) {
            sharedDir = Paths.get(args[0]).toAbsolutePath();
        } else {
            sharedDir = Paths.get(System.getProperty("user.dir"))
                             .getParent().resolve("shared");
        }
        Files.createDirectories(sharedDir);

        char[] passphrase = "Mult1m0d@lS3cur1ty#2025!".toCharArray();

        // Create sample files if shared dir is empty of .txt/.cfg/.csv
        List<Path> sourceFiles = new ArrayList<>();
        String[][] samples = {
            {"layer4_doc.txt",  "Java Layer 4 — AES-256-GCM encryption demonstration\n"},
            {"layer4_meta.cfg", "[layer]\nid=4\nlang=Java\nalgo=AES-256-GCM\nkdf=PBKDF2\n"},
            {"layer4_data.csv", "field,value,hash\nalpha,001,aa11\nbeta,002,bb22\ngamma,003,cc33\n"},
        };
        for (String[] s : samples) {
            Path p = sharedDir.resolve(s[0]);
            if (!Files.exists(p)) Files.writeString(p, s[1]);
            sourceFiles.add(p);
        }
        // Include cross-language files if present
        for (String name : new String[]{"secret_doc.txt", "config.cfg", "data.csv"}) {
            Path p = sharedDir.resolve(name);
            if (Files.exists(p)) sourceFiles.add(p);
        }

        Path zipPath    = sharedDir.resolve("archive_java.zip");
        Path blobPath   = sharedDir.resolve("archive_java.mmsec");
        Path verifyZip  = sharedDir.resolve("archive_java_verify.zip");
        Path xmlPath    = sharedDir.resolve("manifest_java.xml");

        System.out.println("\n[STEP 1] Creating ZIP archive …");
        Map<String, String> fileHashes = createFinalZip(sourceFiles, zipPath);

        System.out.println("\n[STEP 2] Reading ZIP bytes …");
        byte[] zipBytes  = Files.readAllBytes(zipPath);
        String zipSHA256 = sha256hex(zipBytes);
        System.out.println("  ZIP SHA-256: " + zipSHA256);

        System.out.println("\n[STEP 3] AES-256-GCM encrypting …");
        EncResult enc = encryptAESGCM(zipBytes, passphrase);
        System.out.println("  Salt:    " + bytesToHex(enc.salt).substring(0, 16) + "…");
        System.out.println("  IV:      " + bytesToHex(enc.iv));
        System.out.println("  AuthTag: " + bytesToHex(enc.tag));
        System.out.println("  Enc SHA256: " + enc.sha256);

        System.out.println("\n[STEP 4] Writing binary blob …");
        writeBinaryBlob(blobPath, enc);
        String blobSHA256 = sha256file(blobPath);
        System.out.println("  Blob SHA-256: " + blobSHA256);

        System.out.println("\n[STEP 5] Computing HMAC-SHA256 signature …");
        byte[] hmacKey   = Arrays.copyOf(enc.salt, 32);
        String hmacSig   = hmacSHA256(hmacKey, Files.readAllBytes(blobPath));
        System.out.println("  HMAC: " + hmacSig.substring(0, 32) + "…");

        System.out.println("\n[STEP 6] Writing XML manifest …");
        writeXMLManifest(xmlPath, fileHashes, enc, zipSHA256, blobSHA256, hmacSig);

        System.out.println("\n[STEP 7] Verifying round-trip decryption …");
        boolean ok = verifyRoundTrip(blobPath, verifyZip, passphrase, zipSHA256);

        System.out.println("\n[SUMMARY]");
        System.out.println("─────────────────────────────────────────────────────");
        System.out.printf("  Files packed    : %d%n", sourceFiles.size());
        System.out.printf("  ZIP size        : %d bytes%n", Files.size(zipPath));
        System.out.printf("  Blob size       : %d bytes%n", Files.size(blobPath));
        System.out.printf("  Integrity check : %s%n", ok ? "✓ PASSED" : "✗ FAILED");
        System.out.printf("  Cipher          : AES-256-GCM (authenticated)%n");
        System.out.printf("  KDF             : PBKDF2-HMAC-SHA256 (%,d iter)%n", PBKDF2_ITER);
        System.out.println("─────────────────────────────────────────────────────");
        System.out.println("\n[DONE] Java layer complete.");
    }
}
