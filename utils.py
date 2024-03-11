def write_f_strings_to_file(file_path, f_string):
    try:
        with open(file_path, 'a') as file:
            file.write(f"{f_string}\n")
        print("F-Strings wurden erfolgreich in die Datei geschrieben.")
    except Exception as e:
        print(f"Fehler beim Schreiben der F-Strings in die Datei: {e}")


