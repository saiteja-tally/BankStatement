from src.bank_parser import BankStatementParser
import os
import traceback

parser = BankStatementParser()

input = "../samples/bordered"

output = "./output/bordered"

if input.lower().endswith(".pdf"):
    print(f"Processing {os.path.basename(input)}...")
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
                traceback.print_exc()
                errfile = os.path.join(output, "errors.txt")
                os.makedirs(os.path.dirname(errfile), exist_ok=True)
                with open(errfile, "a", encoding="utf-8") as f:
                    f.write(f"Error processing {filepath}: {e}\n")
                    f.write(traceback.format_exc())
                print(f"Error processing {filepath}: {e}")

