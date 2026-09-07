import csv
import random
import string
import base64

# ============ Tablas estándar de DES (FIPS 46-3, dominio público) ============

IP = [58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,
      62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,
      57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,
      61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7]

FP = [40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,
      38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,
      36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,
      34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25]

E = [32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,
     12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,
     24,25,26,27,28,29,28,29,30,31,32,1]

P = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,
     2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]

PC1 = [57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,
       59,51,43,35,27,19,11,3,60,52,44,36,63,55,47,39,
       31,23,15,7,62,54,46,38,30,22,14,6,61,53,45,37,
       29,21,13,5,28,20,12,4]

PC2 = [14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,26,8,
       16,7,27,20,13,2,41,52,31,37,47,55,30,40,51,45,33,48,
       44,49,39,56,34,53,46,42,50,36,29,32]

SHIFTS = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

SBOXES = [
    [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
     0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
     4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
     15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13],
    [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
     3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
     0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
     13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9],
    [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
     13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
     13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
     1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12],
    [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
     13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
     10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
     3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14],
    [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
     14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
     4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
     11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3],
    [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11,
     10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
     9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
     4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13],
    [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
     13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
     1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
     6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12],
    [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
     1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
     7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
     2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11],
]

# ============ Utilidades de bits ============

def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)

def permute(bits, table):
    return [bits[i - 1] for i in table]

# ============ Key schedule ============

def generate_subkeys(key_bytes, num_rounds):
    key_bits = bytes_to_bits(key_bytes)      # 64 bits
    permuted = permute(key_bits, PC1)        # 56 bits
    C, D = permuted[:28], permuted[28:]
    subkeys = []
    for i in range(num_rounds):
        shift = SHIFTS[i % len(SHIFTS)]
        C = C[shift:] + C[:shift]
        D = D[shift:] + D[:shift]
        subkeys.append(permute(C + D, PC2))  # 48 bits
    return subkeys

# ============ Función de Feistel ============

def f_function(R, subkey):
    expanded = permute(R, E)                 # 32 -> 48 bits
    xored = [expanded[i] ^ subkey[i] for i in range(48)]
    sbox_out = []
    for i in range(8):
        block = xored[i*6:(i+1)*6]
        row = (block[0] << 1) | block[5]
        col = (block[1] << 3) | (block[2] << 2) | (block[3] << 1) | block[4]
        val = SBOXES[i][row * 16 + col]
        sbox_out += [(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1]
    return permute(sbox_out, P)              # 32 bits

# ============ DES de N rondas, un bloque de 8 bytes ============

def des_encrypt_block(block_bytes, subkeys):
    bits = permute(bytes_to_bits(block_bytes), IP)
    L, R = bits[:32], bits[32:]
    for subkey in subkeys:
        f_out = f_function(R, subkey)
        L, R = R, [L[i] ^ f_out[i] for i in range(32)]
    cipher_bits = permute(R + L, FP)          # swap final + FP
    return bits_to_bytes(cipher_bits)

# ============ Padding PKCS7 y modo ECB ============

def pkcs7_pad(data, block_size=8):
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def des_encrypt(key_str, plaintext_str, num_rounds=1):
    key_bytes = key_str.encode("latin-1")[:8].ljust(8, b'\x00')
    subkeys = generate_subkeys(key_bytes, num_rounds)

    data = pkcs7_pad(plaintext_str.encode("latin-1"))
    out = bytearray()
    for i in range(0, len(data), 8):
        block = data[i:i+8]
        out += des_encrypt_block(block, subkeys)
    return bytes(out)

# ============ Generador de dataset (misma interfaz que RC4) ============

def random_string(min_len, max_len):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_csv(csv_path, n, min_len, max_len, key_str, num_rounds=1):
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writerow(["plain", "encrypted"])
        for _ in range(n):
            plain = random_string(min_len, max_len)
            encrypted_bytes = des_encrypt(key_str, plain, num_rounds=num_rounds)
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode("ascii")
            writer.writerow([plain, encrypted_b64])

    print(f"CSV file {csv_path} generated with {n} records "
          f"(DES, {num_rounds} ronda(s)). Min length: {min_len}, Max length: {max_len}")
