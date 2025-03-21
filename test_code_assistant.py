import requests
import json

# Test configuration
BASE_URL = "http://127.0.0.1:8001"
CODE_ASSISTANT_ENDPOINT = f"{BASE_URL}/code-assistant"

def test_code_analysis():
    # Sample code for analysis
    code = """
    def calculate_factorial(n):
        if n < 0:
            return None
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    """
    
    payload = {
        "code": code,
        "task_type": "Code Analysis",
        "language": "python"
    }
    
    try:
        response = requests.post(CODE_ASSISTANT_ENDPOINT, json=payload)
        response.raise_for_status()
        print("\n=== Code Analysis Test ===\n")
        print(response.json())
    except Exception as e:
        print(f"Error in code analysis test: {str(e)}")

def test_code_generation():
    # Code generation prompt
    description = """
    Create a function that takes a list of numbers and returns:
    1. The sum of all numbers
    2. The average of the numbers
    3. The minimum and maximum values
    Use proper error handling and type checking.
    """
    
    payload = {
        "code": description,
        "task_type": "Code Generation",
        "language": "python"
    }
    
    try:
        response = requests.post(CODE_ASSISTANT_ENDPOINT, json=payload)
        response.raise_for_status()
        print("\n=== Code Generation Test ===\n")
        print(response.json())
    except Exception as e:
        print(f"Error in code generation test: {str(e)}")

def test_code_optimization():
    # Sample code for optimization
    code = """
    def find_duplicates(numbers):
        duplicates = []
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                    duplicates.append(numbers[i])
        return duplicates
    """
    
    payload = {
        "code": code,
        "task_type": "Code Optimization",
        "language": "python"
    }
    
    try:
        response = requests.post(CODE_ASSISTANT_ENDPOINT, json=payload)
        response.raise_for_status()
        print("\n=== Code Optimization Test ===\n")
        print(response.json())
    except Exception as e:
        print(f"Error in code optimization test: {str(e)}")

def test_bug_finding():
    # Sample code with bugs
    code = """
    def divide_numbers(a, b):
        return a / b
    
    def process_list(numbers):
        result = []
        for i in range(len(numbers)):
            result.append(numbers[i+1])
        return result
    """
    
    payload = {
        "code": code,
        "task_type": "Bug Finding",
        "language": "python"
    }
    
    try:
        response = requests.post(CODE_ASSISTANT_ENDPOINT, json=payload)
        response.raise_for_status()
        print("\n=== Bug Finding Test ===\n")
        print(response.json())
    except Exception as e:
        print(f"Error in bug finding test: {str(e)}")

def run_all_tests():
    print("Starting Code Assistant API Tests...\n")
    test_code_analysis()
    test_code_generation()
    test_code_optimization()
    test_bug_finding()
    print("\nAll tests completed!")

if __name__ == "__main__":
    run_all_tests()