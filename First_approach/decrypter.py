import tensorflow as tf
import numpy as np
import random
import string
from sklearn.preprocessing import MinMaxScaler

# Configuramos el escalador para el rango conocido: 
# Mínimo = 1 (para el padding) y máximo = 122 (para la 'z')
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.array([[1], [122]]))  # Fijamos el rango del escalador

# Función para cifrar con el algoritmo César
def cesar_encrypt(text, shift):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            shift_amount = shift % 26
            # Convertimos a minúsculas para mantener la consistencia
            new_char = chr(((ord(char.lower()) - ord('a') + shift_amount) % 26) + ord('a'))
            encrypted_text += new_char
        else:
            encrypted_text += char
    return encrypted_text

# Función para normalizar la longitud del texto a max_length
def normalize_text(text, max_length, pad_char="_"):
    if len(text) < max_length:
        return text + pad_char * (max_length - len(text))
    return text[:max_length]

# Generador de claves utilizando p y q como límites de la longitud
def generate_cesar_keys(n, p, q):
    print(f"Generando {n} claves con longitudes entre {p} y {q}...")
    keys = []
    for _ in range(n):
        length = random.randint(p, q)  # Longitud aleatoria entre p y q
        plaintext = ''.join(random.choices(string.ascii_lowercase, k=length))
        shift = random.randint(1, 25)    # Desplazamiento aleatorio entre 1 y 25
        encrypted_text = cesar_encrypt(plaintext, shift)
        normalized_encrypted = normalize_text(encrypted_text, q)
        keys.append((plaintext, normalized_encrypted, shift))
    print(f"Se generaron {n} muestras de claves.")
    return keys

# Función para convertir texto en un tensor utilizando códigos ASCII 
# y normalizando al rango [-1, 1] mediante MinMaxScaler.
# Si el texto es menor que 'max_length', se completa con el valor del padding (definido como 1).
def text_to_tensor(text, max_length, pad_char="_"):
    ascii_codes = []
    for char in text:
        if char == pad_char:
            ascii_codes.append(1)  # Valor especial para padding
        else:
            ascii_codes.append(ord(char))
    # Completar si el texto es demasiado corto (aunque en teoría ya lo normalizamos)
    if len(ascii_codes) < max_length:
        ascii_codes.extend([1] * (max_length - len(ascii_codes)))
    ascii_tensor = np.array(ascii_codes, dtype=np.float32).reshape(-1, 1)
    # Aplicar el escalador para normalizar al rango [-1, 1]
    normalized_tensor = scaler.transform(ascii_tensor).flatten()
    return normalized_tensor

# Parámetros de entrada definidos por el usuario
N = 1000  # Número de muestras para entrenamiento
P = 5     # Longitud mínima de la clave
Q = 10    # Longitud máxima (se usará para normalizar los textos)

print("Preparando datos de entrenamiento...")
dataset = generate_cesar_keys(N, P, Q)

# Preparación de datos:
# X es un array donde cada elemento es un vector (de tamaño Q) obtenido al convertir y normalizar el texto cifrado.
# y corresponde al desplazamiento (shift) utilizado.
X = np.array([text_to_tensor(sample[1], Q) for sample in dataset])
y = np.array([sample[2] for sample in dataset])
print(f"Datos preparados: {X.shape[0]} muestras con vector de entrada de tamaño {Q}.")

# Construcción del modelo de la red neuronal en TensorFlow/Keras
print("Construyendo el modelo de la red neuronal...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(Q,)),  # La entrada es un vector de tamaño Q
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)  # Capa de salida para predecir el desplazamiento
])
print("Modelo creado exitosamente.")

# Compilación del modelo
print("Compilando el modelo...")
model.compile(optimizer="adam", loss="mse")
print("Modelo compilado.")

# Entrenamiento del modelo
print("Iniciando entrenamiento de la red neuronal...")
model.fit(X, y, epochs=200, batch_size=32, verbose=1)
print("Entrenamiento completado.")

# Prueba de predicción: se cifra el texto "hello" con un desplazamiento conocido
test_plain = "hello"       # Texto original de prueba
test_shift = 5             # Desplazamiento de prueba
test_encrypted = cesar_encrypt(test_plain, test_shift)
test_normalized = normalize_text(test_encrypted, Q)
test_tensor = text_to_tensor(test_normalized, Q)

predicted_shift = model.predict(test_tensor[np.newaxis, ...]).item()

print("\n--- Prueba de predicción ---")
print(f"Texto plano: {test_plain}")
print(f"Texto cifrado (normalizado a longitud {Q}): {test_normalized}")
print(f"Desplazamiento real: {test_shift}")
print(f"Desplazamiento predicho (red neuronal): {round(predicted_shift)}")