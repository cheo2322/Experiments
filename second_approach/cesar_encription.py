import csv
import random
import string

# --- Configuración ---
NUM_WORDS = 10000
SHIFT = 3  # Desplazamiento del cifrado César
OUTPUT_FILE = "second_approach/cesar_dataset.csv"

# --- Generador de palabras aleatorias ---
def generate_random_word(min_len=4, max_len=10):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# --- Cifrado César ---
def caesar_encrypt(text, shift):
    encrypted = []
    for char in text:
        if char.isalpha():
            base = ord('a')
            encrypted_char = chr((ord(char) - base + shift) % 26 + base)
            encrypted.append(encrypted_char)
        else:
            encrypted.append(char)
    return ''.join(encrypted)

# --- Generación del dataset ---
def generate_dataset(num_words, shift):
    dataset = []
    for _ in range(num_words):
        plain = generate_random_word()
        encrypted = caesar_encrypt(plain, shift)
        dataset.append((plain, encrypted))
    return dataset

# --- Escritura en CSV ---
def save_to_csv(dataset, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['plain_text', 'caesar_encrypted'])
        writer.writerows(dataset)

# --- Ejecución ---
if __name__ == "__main__":
    dataset = generate_dataset(NUM_WORDS, SHIFT)
    save_to_csv(dataset, OUTPUT_FILE)
    print(f"Dataset generado y guardado en '{OUTPUT_FILE}' con {NUM_WORDS} palabras.")