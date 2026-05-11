import yaml
from t1_third_approach.data import rc4
from dataloader.dataloader_seq_2_seq import get_dataloader

def main():
    # Leer configuración
    with open("t1_third_approach/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Extraer parámetros de la sección 'data'
    csv_path = config["data"].get("csv_path")
    min_len = config["data"].get("min_len", 5)
    max_len = config["data"].get("max_len", 12)
    n = config["data"].get("n", 10)
    key_str = config["data"].get("key", "secretkey")

    # Llamar al generador de datos
    rc4.generate_csv(csv_path, n, min_len, max_len, key_str)
    
    # Llamar al dataloader
    data = get_dataloader(csv_path)
    print(f"Primer batch de datos (plain, encrypted): {next(iter(data))}")

if __name__ == "__main__":
    main()
