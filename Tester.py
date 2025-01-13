from StorageHandler import query_faiss
from Main import index, metadata, chunks, llm_answer_to_promt
import google.generativeai as genai

genai.configure(api_key="AIzaSyChjMzI1Pj2EhhJysR4hSI9IvCB1Aa1k8M")

template = """
Another AI is asked a question and answers: "{given_answer}"
Determine if that is close enough to "{expected_answer}", to be considered the same answer.
If you determine that is is in fact a simular enough answer say correct, otherwise say false
ONLY ANSWER CORRECT OR FALSE.
"""

model = genai.GenerativeModel("gemini-1.5-flash")


def unit_test(tst_questions):


    for question in tst_questions:
        knowledge, meta = query_faiss(index, question['t_question'], metadata, chunks)
        answer = llm_answer_to_promt("", question['t_question'], knowledge)

        prompt = template.format(
            expected_answer=question['t_answer'],
            given_answer=answer,
        )

        # Use the model to generate content
        response = str(model.generate_content(prompt))

        # Find the position of the keyword
        position = response.find("text") + 6
        result = response[position:]

        # Find the position of the first closing curly brace and slice the string
        end_position = result.find("\\n")  # Include the closing brace
        result = result[:end_position]

        print(f"question {question['t_question']} \n given answer was {answer}\n answer is: {result} \n")


test_questions = [
    {   't_question': 'How big is the moon?',
        't_answer': 'I do not have sufficient information to answer this question'    },

    {   't_question': 'Who invented "Magic The Gathering"',
        't_answer': 'I do not have sufficient information to answer this question'    },

    {   't_question': 'Who is Zeus?',
        't_answer': 'The supreme ruler of the Gods'    },

    {   't_question': 'Who is ARTEMIS?',
        't_answer': 'Apollo\'s twin sister and daughter of Zeus'    },

]

if __name__ == "__main__":
    unit_test(test_questions)