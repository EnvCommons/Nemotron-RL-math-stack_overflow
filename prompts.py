"""Grading prompt template for Nemotron-RL-Math-Stack-Overflow evaluation."""

MATH_GRADER_TEMPLATE = """You are evaluating a mathematical answer against a reference answer.

# Reference Answer:
{reference_answer}

# Student Answer:
{student_answer}

# Grading Instructions:

Your task is to determine if the student's answer is **mathematically equivalent** to the reference answer, even if formatted differently.

**Grade as CORRECT if:**
- The numerical value is mathematically equivalent (e.g., 5/9 = 0.555... = 0.5556)
- Different but equivalent representations (e.g., sqrt(2) ≈ 1.414, 2*pi ≈ 6.283)
- Equivalent fractions (e.g., 1/2 = 2/4 = 0.5)
- Simplified vs non-simplified forms (e.g., 6/9 = 2/3)
- Boxed notation is ignored (e.g., \\boxed{{42}} = 42)
- Different notation for same answer (e.g., "(1, 2)" = "x=1, y=2" for coordinate problems)
- Equivalent mathematical expressions (e.g., "x^2 - 1" = "(x-1)(x+1)")

**Grade as INCORRECT if:**
- The numerical values are different
- The answer is off by more than reasonable rounding (≥0.01 for most problems)
- Wrong sign (positive vs negative)
- Different mathematical object (e.g., giving a number when answer should be a set)
- Units don't match (if units are specified)

**Important Notes:**
- Ignore formatting differences (spaces, parentheses, LaTeX notation)
- Accept reasonable rounding (e.g., 0.333 for 1/3, 1.414 for sqrt(2))
- Multiple equivalent forms are all correct
- Focus on mathematical correctness, not notation
- If the student provides additional correct work/explanation along with the right answer, grade as CORRECT

**Examples:**
- Reference: "5/9", Student: "0.556" → CORRECT (reasonable rounding)
- Reference: "42", Student: "\\boxed{{42}}" → CORRECT (same value)
- Reference: "1/2", Student: "2/4" → CORRECT (equivalent fractions)
- Reference: "42", Student: "43" → INCORRECT (different values)
- Reference: "sqrt(2)", Student: "1.41" → CORRECT (reasonable approximation)

# Response Format:

First, provide your reasoning in a <reasoning> tag explaining your thought process.
Then, provide your final grade in an <answer> tag with either "CORRECT" or "INCORRECT".

Example:
<reasoning>
The reference answer is 5/9 which equals approximately 0.5556. The student answer is 0.556, which is very close and represents reasonable rounding. These are mathematically equivalent.
</reasoning>
<answer>CORRECT</answer>

Now grade the student's answer.""".strip()
