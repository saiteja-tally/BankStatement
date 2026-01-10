from src.bank_parser import BankStatementParser

parser = BankStatementParser()

input = "../samples/bordered"

output = "./output/bordered"

if input.lower().endswith(".pdf"):
    parser.parse(input, output, password=None)
else:
    import os
    for filename in os.listdir(input):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(input, filename)
            try:
                print(f"Processing {filepath}...")
                parser.parse(filepath, output, password=None)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

