import argparse
import subprocess
import sys
import os

def run_step(step_num):
    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }
    
    if step_num not in steps:
        print(f"Error: Step {step_num} not found.")
        return
        
    # Point to pseudocode folder
    script_path = os.path.join("pseudocode", steps[step_num])
    
    print(f"\n>>> Running Step {step_num}: {script_path} <<<\n")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed with exit code {result.returncode}")
    else:
        print(f"\n[SUCCESS] Step {step_num} completed.")

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab Steps from pseudocode folder")
    parser.add_argument("--step", type=int, help="Specific step to run (1-4)")
    args = parser.parse_args()
    
    if args.step:
        run_step(args.step)
    else:
        print("Running all steps sequentially from pseudocode folder...")
        for i in range(1, 5):
            run_step(i)
            print("-" * 60)

if __name__ == "__main__":
    main()
