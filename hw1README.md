INFO5940 

The program runs on Streamlit by submitting files for analysis. Users can submit files in either "txt" or "pdf" formats and can ask questions regarding the files.

Users may input their own information in the environment section of the docker-compose.yml and may install or change packages in the Dockerfile.

Things users should notice:

The app is designed to answer questions with concise sentences, so no more than three sentences will be generated.
If users ask a question and get "I don't know" as an answer, it could be either because the material isn't covered or, in an unlikely case, doesn't match the exact wording. Users may try to paraphrase the question.
