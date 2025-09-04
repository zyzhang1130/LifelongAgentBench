#!/usr/bin/env python3
"""
Direct model generation test to compare with training behavior
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Same model as training
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Load model exactly like eval would
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

# Test prompt (from your logs)
test_prompt = """User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.

To do operation, the format should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
I will use "Action: Operation" literally to match your SQL.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If the SQL is not executed successfully, the response will be the error message.
Otherwise, the response will be the raw MySQL response.
For SELECT queries, the response will be the result of the query, such as [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...], where each tuple represents a row and the elements are the values of the columns in the row.
For SQL such as INSERT, UPDATE, and DELETE, the response will be empty list [] indicating that the SQL was executed successfully.

If you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:
Action: Answer
Final Answer: [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If the question is about querying the database, then after done operation, your answer field should be the result of the query.
We note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Once you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.

Now, I will give you the question that you need to solve.

Agent: OK.

User: Insert a new payment record for member ID 102 with a payment date of '2023-10-15', amount 75, and payment method 'Credit Card' into the membership payments table.
Table information: {"name": "membership_payments", "row_list": [[1, 101, "2023-01-05", 50, "Cash"], [2, 102, "2023-02-10", 60, "Debit Card"], [3, 103, "2023-03-15", 80, "Credit Card"], [4, 104, "2023-04-20", 70, "Bank Transfer"], [5, 102, "2023-05-25", 65, "Debit Card"], [6, 105, "2023-06-30", 90, "Credit Card"], [7, 106, "2023-07-12", 55, "Cash"], [8, 107, "2023-08-18", 85, "Online Payment"], [9, 108, "2023-09-22", 95, "Check"], [10, 109, "2023-10-01", 60, "Debit Card"], [11, 110, "2023-11-05", 75, "Credit Card"], [12, 111, "2023-12-10", 100, "Bank Transfer"], [13, 112, "2023-01-15", 50, "Cash"], [14, 113, "2023-02-20", 70, "Debit Card"], [15, 114, "2023-03-25", 80, "Credit Card"], [16, 115, "2023-04-30", 65, "Online Payment"], [17, 116, "2023-05-05", 55, "Check"], [18, 117, "2023-06-10", 85, "Cash"], [19, 118, "2023-07-15", 90, "Debit Card"], [20, 119, "2023-08-20", 75, "Credit Card"], [21, 120, "2023-09-25", 60, "Bank Transfer"], [22, 121, "2023-10-05", 95, "Online Payment"], [23, 122, "2023-11-10", 70, "Check"], [24, 123, "2023-12-15", 80, "Cash"], [25, 124, "2023-01-20", 65, "Debit Card"]], "column_info_list": [{"name": "payment_id", "type": "INT"}, {"name": "member_id", "type": "INT"}, {"name": "payment_date", "type": "TEXT"}, {"name": "amount", "type": "INT"}, {"name": "payment_method", "type": "TEXT"}]}"""

print("=" * 80)
print("🧪 TESTING DIRECT MODEL GENERATION")
print("=" * 80)

# Test different generation parameters
test_configs = [
    {"temperature": 0.3, "max_new_tokens": 512, "do_sample": True},
    {"temperature": 0.7, "max_new_tokens": 512, "do_sample": True},
    {"temperature": 0.0, "max_new_tokens": 512, "do_sample": False},  # Greedy
]

for i, gen_config in enumerate(test_configs):
    print(f"\n{'='*50}")
    print(f"🔄 Test {i+1}: {gen_config}")
    print(f"{'='*50}")

    # Tokenize input
    inputs = tok(test_prompt, return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, pad_token_id=tok.eos_token_id, **gen_config)

    # Decode only the new tokens
    new_tokens = outputs[0][input_length:]
    response = tok.decode(new_tokens, skip_special_tokens=True)

    print(f"Response length: {len(response)} chars")
    print(f"First 500 chars: {repr(response[:500])}")
    print(f"Contains 'Action:': {'Action:' in response}")
    print(f"Contains 'INSERT': {'INSERT' in response.upper()}")

print("\n" + "=" * 80)
print("🏁 Direct model test complete")
print("=" * 80)
