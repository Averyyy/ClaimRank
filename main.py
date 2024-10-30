from llm.llama import process_documents


def main():
    process_documents(
        input_dir='dataset/enwiki20201020_test',
        output_dir='dataset'
    )
    # More steps after


if __name__ == "__main__":
    main()
