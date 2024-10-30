from llm.llama import process_documents
import asyncio


def main():
    asyncio.run(process_documents(
        input_dir='dataset/enwiki20201020',
        output_dir='dataset'
    ))
    # More steps after


if __name__ == "__main__":
    main()
