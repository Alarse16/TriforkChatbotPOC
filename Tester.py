from StorageHandler import query_faiss
from Main import index, metadata, chunks, llm_answer_to_promt
import google.generativeai as genai

# Configure the Generative AI API key
# This enables communication with the Generative AI model.
genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")

# Template for validating answers provided by another AI
template = """
Another AI is asked a question and answers: "{given_answer}"
Determine if that is close enough to "{expected_answer}", to be considered the same answer.
If you determine that it is in fact a similar enough answer say correct, otherwise say false
ONLY ANSWER CORRECT OR FALSE.
"""

# Initialize the Generative AI model
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to run unit tests on a set of test questions
def unit_test(tst_questions):
    """
    Validate the AI's answers to a series of test questions.

    Args:
        tst_questions (list): A list of dictionaries with 't_question' and 't_answer'.
            't_question' (str): The test question.
            't_answer' (str): The expected correct answer.
    """
    for question in tst_questions:
        # Retrieve knowledge and metadata for the test question
        knowledge, meta = query_faiss(index, question['t_question'], metadata, chunks)

        # Generate an answer using the LLM
        answer = llm_answer_to_promt("", question['t_question'], knowledge)

        # Format the prompt to validate the generated answer
        prompt = template.format(
            expected_answer=question['t_answer'],
            given_answer=answer,
        )

        # Use the model to generate content
        response = str(model.generate_content(prompt))

        # Extract the result from the response
        position = response.find("text") + 6
        result = response[position:]
        end_position = result.find("\\n")  # Locate the newline character
        result = result[:end_position]

        # Print the results of the unit test
        print(f"Question: {question['t_question']}\nGiven Answer: {answer}\nValidation Result: {result}\n")

# List of test questions and their expected answers
test_questions = [
    {
        't_question': 'How big is the moon?',
        't_answer': 'I do not have sufficient information to answer this question'
    },
    {
        't_question': 'Who invented "Magic The Gathering"',
        't_answer': 'I do not have sufficient information to answer this question'
    },
    {
        't_question': 'Who is Zeus?',
        't_answer': 'The supreme ruler of the Gods'
    },
    {
        't_question': 'Who is ARTEMIS?',
        't_answer': "Apollo's twin sister and daughter of Zeus"
    },
]

# Main execution entry point
if __name__ == "__main__":
    unit_test(test_questions)
