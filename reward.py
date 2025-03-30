import re
import random

from math_verify import parse, verify

from utils.parser import extract_answer as qwen_extract_answer


def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False

def hf_verify_with_try(gold, target):
    try:
        parsed_target = parse(target)    
        parsed_gold = parse(gold)
        return verify(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False
        
def compute_score(solution_str, ground_truth, method='strict'):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    try:
        extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
        
        
        if "\\boxed" not in extract_answer:
            boxed_answer = f"\\boxed{{{extract_answer}}}"
        else:
            boxed_answer = extract_answer
        
        if "\\boxed" not in ground_truth:
            boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
        else:
            boxed_ground_truth = ground_truth
            
        
        # target = parse(boxed_answer)    
        # gold = parse(boxed_ground_truth)
        correct = hf_verify_with_try(gold=boxed_ground_truth, target=boxed_answer)
        
        if correct:
            box_match = 1.0
        else:
            box_match = 0

    except Exception as e:
        print(f"Error: {str(e)}")
        box_match = 0
        correct = False

    # if correct:
    #     # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
    #     print(f"\n[Model Response]\n{solution_str}")
    #     print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
    #     print(f"\n[Extracted Answer]\n{extract_answer}")
    #     print(f"\n[Ground Truth]\n{ground_truth}")
    #     print(f"\n[Reward Score]\n{box_match}")
    #     print(f"\n[Correctness]\n{float(correct)}")

    return {"score": box_match, "correctness": correct}


if __name__ == "__main__":
    # Test case with a solution that contains a boxed answer
    solution_str = """<|im_start|>user
A rectangular garden is 10 meters long and 6 meters wide. The owner wants to create a 1-meter wide path around the garden. What is the area of the path in square meters? Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
<|im_start|>assistant
To find the area of the path, I need to calculate the difference between the area of the outer rectangle and the area of the garden.

The garden is 10 meters long and 6 meters wide.

The outer rectangle includes the garden plus the 1-meter wide path on all sides.
- The length of the outer rectangle is 10 + 1 + 1 = 12 meters
- The width of the outer rectangle is 6 + 1 + 1 = 8 meters

Area of the outer rectangle = 12 × 8 = 96 square meters
Area of the garden = 10 × 6 = 60 square meters

Area of the path = Area of outer rectangle - Area of garden
Area of the path = 96 - 60 = 36 square meters

Therefore, the area of the path is \\boxed{36} square meters.
"""

    ground_truth = "\\boxed{36}"
    
    result = compute_score(solution_str, ground_truth)
    
    print("\n[Test Result]")
    print(f"Score: {result['score']}")
    print(f"Correct: {result['correctness']}")
    
    # Test case with an incorrect solution
    incorrect_solution = """<|im_start|>user
A rectangular garden is 10 meters long and 6 meters wide. The owner wants to create a 1-meter wide path around the garden. What is the area of the path in square meters? Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
<|im_start|>assistant
To find the area of the path, I need to calculate the difference between the area of the outer rectangle and the area of the garden.

The garden is 10 meters long and 6 meters wide.

The outer rectangle includes the garden plus the 1-meter wide path on all sides.
- The length of the outer rectangle is 10 + 1 + 1 = 12 meters
- The width of the outer rectangle is 6 + 1 + 1 = 8 meters

Area of the outer rectangle = 12 × 8 = 96 square meters
Area of the garden = 10 × 6 = 60 square meters

Area of the path = Area of outer rectangle - Area of garden
Area of the path = 96 - 60 = 35 square meters (calculation mistake)

Therefore, the area of the path is \\boxed{35} square meters.
"""
    
    incorrect_result = compute_score(incorrect_solution, ground_truth)
    
    print("\n[Incorrect Test Result]")
    print(f"Score: {incorrect_result['score']}")
    print(f"Correct: {incorrect_result['correctness']}")
