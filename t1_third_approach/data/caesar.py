import csv
import random
import string
import base64

# Implementación sencilla de Cifrado César
def caesar(key, data):
    shift = sum(ord(c) for c in key) % 26  # desplazamiento derivado de la clave
    out = []
    for char in data:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            out.append(chr((ord(char) - base + shift) % 26 + base))
        elif char.isdigit():
            out.append(chr((ord(char) - ord('0') + shift) % 10 + ord('0')))
        else:
            out.append(char)
    return ''.join(out)

def random_string(min_len, max_len):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_csv(csv_path, n, min_len, max_len, key_str):
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writerow(["plain", "encrypted"])
        for _ in range(n):
            plain = random_string(min_len, max_len)
            encrypted = caesar(key_str, plain)
            # Codificar en Base64
            encrypted_b64 = base64.b64encode(encrypted.encode("latin-1")).decode("ascii")
            writer.writerow([plain, encrypted_b64])

    print(f"CSV file {csv_path} generated with {n} records. Min length: {min_len}, Max length: {max_len}")
