import streamlit as st
import os
from sec_edgar_downloader import Downloader
import anthropic
import tiktoken

# Set up Streamlit interface
st.title("10-K Question Answering with Anthropic")

ticker = st.text_input("Enter Firm Ticker")
year = st.number_input("Enter Year", min_value=1990, max_value=2024, value=2023)
question = st.text_area("Ask a question about the firm's 10-K")
anthropic_api_key = st.text_input("Enter your Anthropic API Key", type="password")
model = st.selectbox(
    "Select Anthropic Model",
    ('claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307')
)

# Function to fetch 10-K filing
def fetch_10k_filing(ticker, year):
    """Fetches the 10-K filing for a given ticker and year."""
    try:
        # Initialize the downloader
        # The download directory is set to the current directory
        dl = Downloader()

        # Download the 10-K filing
        # The downloader saves filings to a specific directory structure
        # We will download only one filing to ensure we get the correct one for the year
        dl.get("10-K", ticker, limit=1, after=f"{year}-01-01", before=f"{year}-12-31")

        # Construct the expected file path based on the downloader's structure
        # and find the exact filename within the directory
        download_path = os.path.join("sec-edgar-filings", ticker, "10-K")
        filing_content = ""
        filing_found = False

        if os.path.exists(download_path):
            # Walk through the downloaded directory to find the filing text file
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.endswith(".txt"): # 10-K filings are often in text format
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            filing_content = f.read()
                        filing_found = True
                        break # Assume the first .txt file found is the 10-K
                if filing_found:
                    break # Stop searching once content is found

        if not filing_content:
             return f"Could not find content in downloaded filing for {ticker} in {year}."

        return filing_content

    except Exception as e:
        return f"An error occurred: {e}"

# Function to process 10-K text (chunking)
def process_10k_text(text, model_name="claude-3-haiku-20240307", max_chunk_tokens=4000):
    """
    Processes the raw 10-K text by splitting it into chunks that respect
    the context window limits of the specified language model.

    Args:
        text: The raw text of the 10-K filing.
        model_name: The name of the Anthropic model (used to get encoding).
        max_chunk_tokens: The maximum number of tokens per chunk, leaving
                          room for prompt and answer tokens.

    Returns:
        A list of text chunks.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    current_chunk_tokens = []

    for token in tokens:
        current_chunk_tokens.append(token)
        if len(current_chunk_tokens) >= max_chunk_tokens:
            chunks.append(encoding.decode(current_chunk_tokens))
            current_chunk_tokens = []

    # Add the last chunk if it's not empty
    if current_chunk_tokens:
        chunks.append(encoding.decode(current_chunk_tokens))

    return chunks

# Function to interact with Anthropic API
def get_answer_from_anthropic(api_key, model_name, question, text_chunks):
    """
    Sends the user's question and text chunks from the 10-K to the Anthropic API
    to get an answer.

    Args:
        api_key: The Anthropic API key.
        model_name: The name of the Anthropic model to use.
        question: The user's question about the 10-K.
        text_chunks: A list of text chunks from the processed 10-K.

    Returns:
        A string containing the synthesized answer or an error message.
    """
    client = anthropic.Anthropic(api_key=api_key)
    all_responses = []

    for i, chunk in enumerate(text_chunks):
        try:
            # Construct a prompt that clearly instructs the model
            prompt = f"""You are an AI assistant specifically designed to answer questions based on provided text snippets from a company's 10-K filing.
Read the following text carefully and answer the question based *only* on the information contained within this text.
If the text does not contain enough information to answer the question, state that you cannot answer based on the provided text.

10-K Text Snippet (Chunk {i+1}/{len(text_chunks)}):
---
{chunk}
---

User Question:
{question}

Answer (based ONLY on the provided text snippet):"""

            message = client.messages.create(
                model=model_name,
                max_tokens=1024, # Adjust based on expected answer length
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            all_responses.append(message.content[0].text)

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            all_responses.append(f"Error processing chunk {i+1}: {e}")

    synthesized_answer = "\n\n---\n\n".join(all_responses)

    if not synthesized_answer.strip():
        return "Could not generate an answer based on the provided text."

    if len(text_chunks) > 1 and len([resp for resp in all_responses if "cannot answer" not in resp.lower()]) > 1:
         try:
             synthesis_prompt = f"""You have been provided with several partial answers to a question based on different chunks of a 10-K filing.
             Synthesize these partial answers into a single, coherent, and comprehensive answer to the original question.
             If some partial answers indicate they cannot answer based on their specific chunk, ignore those and focus on the ones that provide relevant information.

             Original Question: {question}

             Partial Answers:
             ---
             {synthesized_answer}
             ---

             Synthesized Final Answer:"""
             synthesis_message = client.messages.create(
                 model=model_name,
                 max_tokens=1024,
                 messages=[
                     {"role": "user", "content": synthesis_prompt}
                 ]
             )
             return synthesis_message.content[0].text
         except Exception as e:
             print(f"Error during synthesis: {e}")
             return f"Generated partial answers but failed to synthesize: {synthesized_answer}\nError: {e}"

    return synthesized_answer


# Combine all parts in the Streamlit app's button logic
if st.button("Get Answer"):
    if not ticker or not question or not anthropic_api_key:
        st.warning("Please fill in Ticker, Question, and Anthropic API Key.")
    else:
        st.info(f"Fetching 10-K for {ticker} ({year})...")
        filing_text = fetch_10k_filing(ticker, year)

        if "Error" in filing_text or "Could not find content" in filing_text:
            st.error(filing_text)
        else:
            st.success("10-K fetched successfully. Processing...")
            processed_chunks = process_10k_text(filing_text, model_name=model)

            if not processed_chunks:
                 st.warning("Could not process the 10-K text into usable chunks.")
            else:
                st.success(f"10-K processed into {len(processed_chunks)} chunks. Getting answer from Anthropic...")

                answer = get_answer_from_anthropic(anthropic_api_key, model, question, processed_chunks)

                if "Error" in answer or "Could not generate an answer" in answer:
                    st.error(answer)
                else:
                    st.success("Answer received!")
                    st.write("## Answer:")
                    st.write(answer)