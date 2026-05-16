import csv
import random
import string

# Implementación sencilla de RC4
def rc4(key, data):
    S = list(range(256))
    j = 0
    out = []

    # KSA (Key Scheduling Algorithm)
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]

    # PRGA (Pseudo-Random Generation Algorithm)
    i = j = 0
    for char in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        out.append(chr(ord(char) ^ K))

    return ''.join(out)

def random_string(min_len, max_len):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_csv(csv_path, n, min_len, max_len, key_str):
    key = [ord(c) for c in key_str]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writerow(["plain", "encrypted"])
        for _ in range(n):
            plain = random_string(min_len, max_len)
            encrypted = rc4(key, plain)
            writer.writerow([plain, encrypted])

    print(f"CSV file {csv_path} generated with {n} records.")
