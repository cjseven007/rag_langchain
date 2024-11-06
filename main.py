import streamlit as st
from document_loader import load_vector_store
from chain import initialize_qa_chain

# Streamlit UI
st.title("RAG-Powered Q&A App with Hugging Face")
st.write("Upload a document and ask any question!")

uploaded_file = st.file_uploader("Upload your text document", type=["pdf"])

if uploaded_file is not None:
    with open("uploaded_doc.txt", "wb") as f:
        f.write(uploaded_file.read())

    vector_store = load_vector_store("uploaded_doc.txt")
    retriever = vector_store.as_retriever()

    # Initialize the QA chain
    qa_chain = initialize_qa_chain(retriever)

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Searching for the answer..."):
            response = qa_chain.run(query)

            # Find the index of "Helpful Answer" and "Question" in the response
            helpful_answer_start = response.find("Helpful Answer")
            question_start = response.find("Question", helpful_answer_start)

            st.success("Answer:")
            if helpful_answer_start != -1:
                if question_start != -1:
                    # Trim the response to include only the part between "Helpful Answer" and "Question"
                    helpful_answer = response[helpful_answer_start:question_start].strip(
                    )
                else:
                    # Include everything after "Helpful Answer" if "Question" is not found
                    helpful_answer = response[helpful_answer_start:].strip()

                st.write(helpful_answer)
            else:
                st.write("No helpful answer found.")

else:
    st.info("Please upload a text document to start.")
