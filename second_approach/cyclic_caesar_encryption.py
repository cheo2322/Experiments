import csv
import random
import string

# --- Configuración ---
NUM_WORDS = 10000
SHIFT_SEQ = [2, 3, 6, 7, 5]  # Secuencia cíclica de desplazamientos
OUTPUT_FILE = "second_approach/cesar_dataset.csv"

# --- Generador de palabras aleatorias ---
def generate_random_word(min_len=4, max_len=10):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# --- Cifrado César con secuencia de shifts ---
def caesar_encrypt_seq(text, shift_seq):
    encrypted = []
    for i, char in enumerate(text):
        if char.isalpha():
            base = ord('a')
            shift = shift_seq[i % len(shift_seq)]
            encrypted_char = chr((ord(char) - base + shift) % 26 + base)
            encrypted.append(encrypted_char)
        else:
            encrypted.append(char)
    return ''.join(encrypted)

# --- Generación del dataset ---
def generate_dataset(num_words, shift_seq):
    dataset = []
    for _ in range(num_words):
        plain = generate_random_word()
        encrypted = caesar_encrypt_seq(plain, shift_seq)
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
    dataset = generate_dataset(NUM_WORDS, SHIFT_SEQ)
    save_to_csv(dataset, OUTPUT_FILE)
    print(f"Dataset generado y guardado en '{OUTPUT_FILE}' con {NUM_WORDS} palabras.")
