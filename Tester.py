from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from StorageHandler import query_faiss
from Main import index, metadata, chunks, llm_answer_to_promt


template = """
Another AI is asked a question and answers: "{given_answer}"
Determine if that is close enough to "{expected_answer}", to be considered the same answer.
If you determine that is is in fact a simular enough answer say correct, otherwise say false
ONLY ANSWER CORRECT OR FALSE.
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def unit_test(tst_questions):
    for question in tst_questions:
        knowledge, meta = query_faiss(index, question['t_question'], metadata, chunks)
        answer = llm_answer_to_promt("", question['t_question'])
        result = chain.invoke({"expected_answer": question['t_answer'],
                               "given_answer": answer
                               })
        print(f"question {question['t_question']} \n given answer was {answer}\n answer is: {result} \n")


test_questions = [
    {   't_question': 'How big is the moon?',
        't_answer': '//insouciant knowledge//'    },

    {   't_question': 'Who invented "Magic The Gathering"',
        't_answer': '//insouciant knowledge//'    },

    {   't_question': 'Who is Zeus?',
        't_answer': 'The supreme ruler of the Gods'    },

    {   't_question': 'Who is ARTEMIS?',
        't_answer': 'Apollo\'s twin sister and daughter of Zeus'    },

]

if __name__ == "__main__":
    unit_test(test_questions)