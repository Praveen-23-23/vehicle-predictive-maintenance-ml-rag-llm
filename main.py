from pipeline.inference_pipeline import run_pipeline

if __name__ == "__main__":
    risk, explanation = run_pipeline()

    print("===============================\n")
    print(explanation)
    print("\n===============================")