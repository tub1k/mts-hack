import zipfile

SUBMIT_FILES = [
    "solution.py",
    "config.py",
    "prompts.py",
    "rag.py",
    "agent.py",
    "requirements.txt",
]

def create_submit(zip_name: str = "submit.zip"):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in SUBMIT_FILES:
            zf.write(file, arcname=file)  # arcname=file гарантирует отсутствие вложенных папок
    print(f"Архив {zip_name} готов к отправке на платформу!")

if __name__ == "__main__":
    create_submit()