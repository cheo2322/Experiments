import random
import string

def cesar_encrypt(text, shift):
    """Función que cifra un texto utilizando el cifrado César con un desplazamiento dado."""
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            shift_amount = shift % 26
            new_char = chr(((ord(char.lower()) - ord('a') + shift_amount) % 26) + ord('a'))
            encrypted_text += new_char.upper() if char.isupper() else new_char
        else:
            encrypted_text += char
    return encrypted_text

def generate_cesar_keys(n, p, q):
    """Genera N claves encriptadas usando cifrado César con longitudes entre P y Q."""
    keys = []
    for _ in range(n):
        length = random.randint(p, q)
        plaintext = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        shift = random.randint(1, 25)  # Desplazamiento aleatorio entre 1 y 25
        encrypted_text = cesar_encrypt(plaintext, shift)
        keys.append((plaintext, encrypted_text, shift))
    return keys

# Parámetros de ejemplo
N = 5  # Número de claves a generar
P = 8  # Longitud mínima de clave
Q = 12 # Longitud máxima de clave

# Generar claves
cesar_keys = generate_cesar_keys(N, P, Q)

# Mostrar claves generadas
for i, (plaintext, encrypted, shift) in enumerate(cesar_keys):
    print(f"Clave {i+1}: Texto plano: {plaintext}, Cifrado: {encrypted}, Desplazamiento: {shift}")