# **RAG-Based Multilingual News Retrieval System**

---

## **Table of Contents**
1. **Project Overview**  
2. **Features**  
3. **Workflow**  
    - **1. Data Collection and Loading**  
    - **2. Data Preprocessing**  
    - **3. Translation**  
    - **4. Summarization**  
    - **5. Model Inference**  
    - **6. Retrieval-Augmented Generation (RAG)**  
    - **7. Similarity and Relevance Analysis**  
    - **8. Streamlit Interactive Application**  
4. **Installation**  
5. **Usage Instructions**  
6. **File Structure**  
7. **Technologies Used**  
8. **Contributors**  

---

## **1. Project Overview**  
The **RAG-Based Multilingual News Retrieval System** enables users to search for **multilingual news articles** and receive **summaries of the top 5 most relevant news articles** in English. This system incorporates **retrieval, translation, and summarization** into a unified pipeline. It leverages **MBART50** for translation, **SBERT embeddings** for similarity, **FAISS indexing** for fast retrieval, and a user-friendly **Streamlit-based UI** for interactivity. The system ensures context-aware summaries of news articles in **German, Spanish, French, Russian, Turkish, and Arabic**.  

---

## **2. Features**  
- **Cross-Lingual Retrieval**: Users submit a query in English and receive the **top 5 most relevant news articles** from multilingual sources.  
- **Multilingual Support**: Supports articles in **German, Spanish, French, Russian, Turkish, and Arabic**.  
- **Translation**: Non-English articles are translated into English using **MBART50**.  
- **Summarization**: Summarized versions of retrieved articles are displayed for users.  
- **Interactive UI**: **Streamlit UI** allows users to submit queries, view results, and analyze similarity scores.  
- **Context-Aware Summaries**: Retrieves and summarizes articles using **RAG**, ensuring relevance and contextual accuracy.  
- **Relevance Analysis**: Measures similarity between the user query and retrieved articles using **BERTScore, Cosine Similarity, and SBERT Score**.  

---

## **3. Workflow**

### **1. Data Collection and Loading**
**Objective**: Collect, load, and prepare multilingual data from the MLSUM dataset.  
**Steps**:
1. **Dataset Selection**: Use the MLSUM dataset for **German, Spanish, French, Russian, and Turkish**.  
2. **Data Download**: **700 records** per language are downloaded from the training set.  
3. **Dataframe Creation**: Data is stored in a Pandas DataFrame with columns:  
   - **Text**: The main content of the article.  
   - **Summary**: The reference summary of the article.  
   - **Language**: The language of the article (e.g., "de" for German).  
4. **Data Concatenation**: Combine all language DataFrames into one unified DataFrame.  
5. **Dataset Conversion**: Convert the DataFrame to a HuggingFace dataset format for compatibility with NLP models.  

---

### **2. Data Preprocessing**
**Objective**: Clean and prepare the text data for translation and summarization.  
**Steps**:
1. **Language Column Addition**: Add a column specifying the language of each article.  
2. **Data Cleaning**: Remove unnecessary columns (e.g., URLs) and irrelevant content.  
3. **Language Detection (Optional)**: Verify that the language tag of each article is correct using **langdetect**.  

---

### **3. Translation**
**Objective**: Translate non-English articles into English using **MBART50**.  
**Steps**:
1. **Model Selection**: Use the **MBART50** model for translation.  
2. **Tokenizer Initialization**: Use **MBart50Tokenizer** with source language (`src_lang`) and target language (`tgt_lang`) set to English (`en_XX`).  
3. **Translation Function**:  
   - Tokenize the input text.  
   - Pass the tokenized input to the **MBART50** model.  
   - Decode the translated output.  
4. **Translation Output**: Store the translated content for summarization.  

---

### **4. Summarization**
**Objective**: Summarize the translated articles using **T5**.  
**Steps**:
1. **Model Selection**: Use **T5** for summarization.  
2. **Summarization Function**:  
   - Tokenize the translated text.  
   - Generate the summary using **T5**.  
   - Decode the summary.  
3. **Summarization Output**: Store the summary for later retrieval and user interaction.  

---

### **5. Model Inference**
**Objective**: Apply **translation and summarization** to all articles in the dataset.  
**Steps**:
1. **Batch Processing**: Translate and summarize each article.  
2. **Result Storage**: Store the translated articles and their summaries for use in the **RAG system**.  

---

### **6. Retrieval-Augmented Generation (RAG)**
**Objective**: Use a **RAG-based retrieval system** to retrieve and summarize news articles based on a user query.  
**Steps**:
1. **Knowledge Base Construction**:  
   - Index all translated articles and summaries using **FAISS**.  
2. **Query Embedding**: Convert user query into an embedding using **SBERT**.  
3. **Information Retrieval**:  
   - Retrieve the **top 5 most relevant articles** from the FAISS index.  
4. **RAG Model**: Generate a **context-aware summary** using the 5 retrieved articles.  
5. **Storage of Results**: Store the retrieved articles, summaries, and RAG-generated output for display in the UI.  

---

### **7. Similarity and Relevance Analysis**
**Objective**: Measure the alignment between the input query and retrieved summaries.  
**Steps**:
1. **BERTScore**: Measures the semantic similarity between the **query and summaries**.  
2. **Cosine Similarity**: Measures similarity using **SBERT embeddings**.  
3. **SBERT Score**: Measures contextual alignment between the user query and system-generated summaries.  

---

### **8. Streamlit Interactive Application**
**Objective**: Allow users to interact with the system via **Streamlit**.  
**Features**:  
1. **Query Submission**: Users submit queries in English.  
2. **Result Display**: Shows **titles, summaries, and similarity scores** for retrieved articles.  
3. **Configurable Settings**: Users can configure dataset path, number of results, and similarity score thresholds.  

---

## **4. Installation**
1. **Install Required Libraries**:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit Application**:  
   ```bash
   streamlit run app.py
   ```

---

## **5. Usage Instructions**
1. **Launch the App**: Run `streamlit run app.py`.  
2. **Input Query**: Enter an English query in the search box.  
3. **View Results**: View the **top 5 most relevant articles**, their summaries, and **similarity scores**.  

---

## **6. File Structure**
```
📦 RAG-Based-Multilingual-News-Retrieval
 ┣ 📂data
 ┣ 📂models
 ┣ 📂notebooks
 ┣ 📜app.py
 ┣ 📜requirements.txt
 ┣ 📜README.md
```

---

## **7. Technologies Used**
- **Python**  
- **HuggingFace Transformers (MBART50, BERT, SBERT, T5)**  
- **FAISS (Dense Vector Indexing)**  
- **Streamlit (Interactive UI)**  
- **SBERT (Sentence Embeddings)**  


