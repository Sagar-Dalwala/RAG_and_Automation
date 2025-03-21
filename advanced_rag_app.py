import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import base64
import requests
from io import BytesIO, StringIO
from streamlit_option_menu import option_menu
from rag_utils import (
    process_input, 
    answer_question, 
    document_metadata,
    qa_history
)

# Page configuration
st.set_page_config(page_title="Advanced RAG System", layout="wide")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
        flex: 1;
        min-width: 200px;
        margin-right: 1rem;
    }
    .source-card {
        background-color: #f1f8e9;
        border-left: 4px solid #7cb342;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.25rem;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f57f17;
        font-weight: bold;
    }
    .confidence-low {
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "current_query" not in st.session_state:
    st.session_state["current_query"] = ""
if "current_answer" not in st.session_state:
    st.session_state["current_answer"] = None
if "visualization_data" not in st.session_state:
    st.session_state["visualization_data"] = None
if "document_count" not in st.session_state:
    st.session_state["document_count"] = 0
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Input"
if "previous_input_type" not in st.session_state:
    st.session_state["previous_input_type"] = None

def get_download_link(data, filename, text):
    """Generate a download link for data"""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_confidence_score(score):
    """Display confidence score with appropriate styling"""
    if score >= 0.8:
        return f'<span class="confidence-high">{score:.2f}</span>'
    elif score >= 0.5:
        return f'<span class="confidence-medium">{score:.2f}</span>'
    else:
        return f'<span class="confidence-low">{score:.2f}</span>'

def main():
    # Main header
    st.markdown('<h1 class="main-header">Advanced RAG Q&A System</h1>', unsafe_allow_html=True)
    
    # Navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Input", "Q&A", "Visualization", "Analytics", "AI Code Review", "Export"],
        icons=["cloud-upload", "chat-dots", "graph-up", "bar-chart", "code-slash", "download"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#6c757d", "font-size": "1rem"},
            "nav-link": {"font-size": "0.9rem", "text-align": "center", "margin": "0px", "padding": "10px"},
            "nav-link-selected": {"background-color": "#7952b3", "color": "white"},
        }
    )
    
    st.session_state["active_tab"] = selected
    
    # Input Tab
    if selected == "Input":
        st.markdown('<h2 class="sub-header">Document Input</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            input_type = st.selectbox("Select Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
            
            # Clear question input if input type changes
            if st.session_state.get("previous_input_type") != input_type:
                if "query_input" in st.session_state:
                    st.session_state["query_input"] = ""
                st.session_state["previous_input_type"] = input_type
            
            # Different input options based on type
            if input_type == "Link":
                number_input = st.number_input("Number of Links", min_value=1, max_value=20, step=1, value=1)
                input_data = []
                for i in range(number_input):
                    url = st.text_input(f"URL {i+1}", key=f"url_{i}")
                    input_data.append(url)
            elif input_type == "Text":
                input_data = st.text_area("Enter Text", height=200)
            else:  # File upload options
                file_types = {"PDF": ["pdf"], "DOCX": ["docx", "doc"], "TXT": ["txt"]}
                input_data = st.file_uploader(f"Upload {input_type} File", type=file_types[input_type])
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
                with col2:
                    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, 10)
                include_metadata = st.checkbox("Include Metadata", value=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                process_button = st.button("Process Document", type="primary")
            with col2:
                clear_button = st.button("Clear Document")
                
            if clear_button:
                st.session_state["vectorstore"] = None
                st.session_state["current_query"] = ""
                st.session_state["current_answer"] = None
                st.session_state["visualization_data"] = None
                st.session_state["document_count"] = 0
                if "query_input" in st.session_state:
                    st.session_state["query_input"] = ""
                st.success("Document cleared successfully!")
                
            if process_button:
                with st.spinner("Processing document..."):
                    try:
                        vectorstore = process_input(
                            input_type, 
                            input_data, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap, 
                            include_metadata=include_metadata
                        )
                        st.session_state["vectorstore"] = vectorstore
                        st.session_state["document_count"] = len(document_metadata)
                        st.success(f"Document processed successfully! Created {len(document_metadata)} chunks.")
                        
                        # Since we've reverted rag_utils.py, we need to skip visualization data generation
                        st.session_state["visualization_data"] = None
                        st.session_state["wordcloud"] = None
                        st.session_state["key_concepts"] = None
                        
                        # Suggest moving to Q&A tab
                        st.info("Document ready for querying! Navigate to the Q&A tab to ask questions.")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Q&A Tab
    elif selected == "Q&A":
        st.markdown('<h2 class="sub-header">Question & Answer</h2>', unsafe_allow_html=True)
        
        if st.session_state["vectorstore"] is None:
            st.warning("Please process a document first in the Input tab.")
        else:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                query = st.text_input("Ask your question", key="query_input")
                col1, col2 = st.columns([1, 1])
                with col1:
                    include_sources = st.checkbox("Include Sources", value=True)
                with col2:
                    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
                
                if st.button("Submit Question", type="primary"):
                    if query.strip():
                        with st.spinner("Generating answer..."):
                            try:
                                answer = answer_question(
                                    st.session_state["vectorstore"], 
                                    query
                                )
                                # Add missing fields that the UI expects
                                answer = {
                                    "result": answer,
                                    "answer": answer["result"],
                                    "confidence_score": 0.8,
                                    "response_time_seconds": 1.0,
                                    "timestamp": "Now",
                                    "sources": []
                                }
                                st.session_state["current_query"] = query
                                st.session_state["current_answer"] = answer
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display answer if available
            if st.session_state["current_answer"]:
                answer = st.session_state["current_answer"]
                
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Answer")
                    st.write(answer["answer"])
                    
                    # Display metrics
                    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                    
                    # Confidence score
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Confidence Score</h4>
                        <p>{display_confidence_score(answer['confidence_score'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Response time
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Response Time</h4>
                        <p>{answer['response_time_seconds']:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Timestamp
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Timestamp</h4>
                        <p>{answer['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display sources if available
                    if "sources" in answer and answer["sources"]:
                        with st.expander("View Sources", expanded=True):
                            for i, source in enumerate(answer["sources"]):
                                st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                                st.markdown(f"**Source {i+1}**")
                                
                                # Display source metadata
                                if "source" in source:
                                    st.markdown(f"**Source:** {source['source']}")
                                if "page" in source:
                                    st.markdown(f"**Page:** {source['page']}")
                                if "paragraph" in source:
                                    st.markdown(f"**Paragraph:** {source['paragraph']}")
                                if "filename" in source:
                                    st.markdown(f"**File:** {source['filename']}")
                                
                                # Display content snippet
                                st.markdown("**Content:**")
                                st.markdown(f"<p style='font-style: italic;'>{source['content']}</p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Tab
    elif selected == "Visualization":
        st.markdown('<h2 class="sub-header">Document Visualization</h2>', unsafe_allow_html=True)
        
        if st.session_state["vectorstore"] is None:
            st.warning("Please process a document first in the Input tab.")
        else:
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Document Similarity", "Word Cloud", "Key Concepts", "Document Map"])
            
            # Document Similarity Matrix
            with viz_tabs[0]:
                if "visualization_data" in st.session_state and st.session_state["visualization_data"] is not None:
                    viz_data = st.session_state["visualization_data"]
                    similarity_matrix = viz_data["similarity_matrix"]
                    docs = viz_data["docs"]
                    
                    st.markdown("### Document Chunk Similarity Matrix")
                    st.write("This heatmap shows how similar different chunks of your document are to each other.")
                    
                    # Create a heatmap using Plotly
                    fig = px.imshow(
                        similarity_matrix,
                        labels=dict(x="Document Chunk", y="Document Chunk", color="Similarity"),
                        x=[f"Chunk {i+1}" for i in range(len(similarity_matrix))],
                        y=[f"Chunk {i+1}" for i in range(len(similarity_matrix))],
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show chunk content on hover
                    with st.expander("View Document Chunks"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}**")
                            st.markdown(f"<div style='max-height: 200px; overflow-y: auto; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>{doc.page_content[:300]}...</div>", unsafe_allow_html=True)
                else:
                    st.info("Visualization data not available. Try processing your document again.")
            
            # Word Cloud Visualization
            with viz_tabs[1]:
                if "wordcloud" in st.session_state and st.session_state["wordcloud"] is not None:
                    st.markdown("### Word Cloud")
                    st.write("This visualization shows the most frequent terms in your documents.")
                    
                    # Display the word cloud
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(st.session_state["wordcloud"], interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Word cloud data not available. Try processing your document again.")
            
            # Key Concepts Visualization
            with viz_tabs[2]:
                if "key_concepts" in st.session_state and st.session_state["key_concepts"]:
                    st.markdown("### Key Concepts")
                    st.write("This chart shows the most important concepts in your documents based on frequency.")
                    
                    # Create a bar chart of key concepts
                    concepts = st.session_state["key_concepts"]
                    df = pd.DataFrame(list(concepts.items()), columns=["Concept", "Frequency"])
                    df = df.sort_values("Frequency", ascending=False)
                    
                    fig = px.bar(
                        df, 
                        x="Concept", 
                        y="Frequency",
                        color="Frequency",
                        color_continuous_scale="Viridis",
                        title="Key Concepts in Document"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Key concepts data not available. Try processing your document again.")
            
            # Document Map
            with viz_tabs[3]:
                st.markdown("### Document Map")
                st.write("This visualization shows how your document chunks relate to each other in semantic space.")
                
                if "visualization_data" in st.session_state and st.session_state["visualization_data"] is not None:
                    # Create a force-directed graph visualization
                    viz_data = st.session_state["visualization_data"]
                    similarity_matrix = viz_data["similarity_matrix"]
                    
                    # Create a network graph
                    # First, we'll create a simplified 2D projection of the document embeddings
                    from sklearn.manifold import TSNE
                    
                    # Get embeddings from similarity matrix
                    embeddings = []
                    for i in range(len(similarity_matrix)):
                        embeddings.append(similarity_matrix[i])
                    
                    # Create 2D projection
                    tsne = TSNE(n_components=2, random_state=42)
                    try:
                        node_positions = tsne.fit_transform(embeddings)
                        
                        # Create a DataFrame for the nodes
                        nodes_df = pd.DataFrame({
                            "x": node_positions[:, 0],
                            "y": node_positions[:, 1],
                            "name": [f"Chunk {i+1}" for i in range(len(node_positions))],
                            "size": [10] * len(node_positions)
                        })
                        
                        # Create a scatter plot
                        fig = px.scatter(
                            nodes_df, x="x", y="y", text="name", size="size",
                            color_discrete_sequence=["#6200ea"],
                            title="Document Semantic Map"
                        )
                        
                        # Add edges between similar nodes
                        threshold = 0.5  # Similarity threshold for drawing edges
                        for i in range(len(similarity_matrix)):
                            for j in range(i+1, len(similarity_matrix)):
                                if similarity_matrix[i][j] > threshold:
                                    fig.add_shape(
                                        type="line",
                                        x0=node_positions[i, 0], y0=node_positions[i, 1],
                                        x1=node_positions[j, 0], y1=node_positions[j, 1],
                                        line=dict(color="#e0e0e0", width=0.5 + similarity_matrix[i][j])
                                    )
                        
                        # Update layout
                        fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="DarkSlateGrey")))
                        fig.update_layout(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating document map: {str(e)}")
                else:
                    st.info("Document map data not available. Try processing your document again.")
    
    # Analytics Tab
    elif selected == "Analytics":
        st.markdown('<h2 class="sub-header">Q&A Analytics</h2>', unsafe_allow_html=True)
        
        if not qa_history:
            st.warning("No Q&A history available yet. Ask some questions in the Q&A tab first.")
        else:
            # Create tabs for different analytics views
            analytics_tabs = st.tabs(["Performance Metrics", "Question Analysis", "Source Usage"])
            
            # Performance Metrics
            with analytics_tabs[0]:
                st.markdown("### Performance Metrics")
                
                # Create a DataFrame from QA history
                qa_df = pd.DataFrame(qa_history)
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Questions", len(qa_df))
                with col2:
                    avg_confidence = qa_df["confidence_score"].mean() if "confidence_score" in qa_df else 0
                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                with col3:
                    avg_response_time = qa_df["response_time_seconds"].mean() if "response_time_seconds" in qa_df else 0
                    st.metric("Average Response Time", f"{avg_response_time:.2f}s")
                
                # Create a line chart of response times
                if "response_time_seconds" in qa_df:
                    st.markdown("#### Response Time Trend")
                    fig = px.line(
                        qa_df, 
                        y="response_time_seconds", 
                        labels={"index": "Question Number", "response_time_seconds": "Response Time (s)"},
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a histogram of confidence scores
                if "confidence_score" in qa_df:
                    st.markdown("#### Confidence Score Distribution")
                    fig = px.histogram(
                        qa_df, 
                        x="confidence_score",
                        nbins=10,
                        color_discrete_sequence=["#7952b3"],
                        labels={"confidence_score": "Confidence Score"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Question Analysis
            with analytics_tabs[1]:
                st.markdown("### Question Analysis")
                
                # Display recent questions
                st.markdown("#### Recent Questions")
                recent_qa = qa_df.tail(5) if len(qa_df) > 5 else qa_df
                for i, row in recent_qa.iterrows():
                    with st.container():
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"**Q: {row['query']}**")
                        st.markdown(f"A: {row['answer'][:200]}..." if len(row['answer']) > 200 else f"A: {row['answer']}")
                        st.markdown(f"Confidence: {display_confidence_score(row['confidence_score'])}", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Question word cloud if we have enough questions
                if len(qa_df) >= 5:
                    st.markdown("#### Question Topics")
                    questions_text = " ".join(qa_df["query"].tolist())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(questions_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
            
            # Source Usage
            with analytics_tabs[2]:
                st.markdown("### Source Usage Analysis")
                
                # Count sources used in answers
                source_counts = {}
                for qa in qa_history:
                    if "sources" in qa:
                        for source in qa["sources"]:
                            if "source" in source:
                                source_name = source["source"]
                                if source_name in source_counts:
                                    source_counts[source_name] += 1
                                else:
                                    source_counts[source_name] = 1
                
                if source_counts:
                    # Create a pie chart of source usage
                    source_df = pd.DataFrame(list(source_counts.items()), columns=["Source", "Count"])
                    fig = px.pie(
                        source_df,
                        values="Count",
                        names="Source",
                        title="Source Usage Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No source usage data available yet.")
    
    # Export Tab
    elif selected == "Export":
        st.markdown('<h2 class="sub-header">Export Session Data</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Export Options")
            
            export_format = st.selectbox("Select Export Format", ["JSON", "CSV", "HTML"])
            include_qa = st.checkbox("Include Q&A History", value=True)
            include_docs = st.checkbox("Include Document Metadata", value=True)
            
            if st.button("Generate Export", type="primary"):
                try:
                    # Prepare data for export
                    export_qa = qa_history if include_qa else []
                    export_docs = document_metadata if include_docs else {}
                    
                    # Generate export based on selected format
                    # Since we've reverted rag_utils.py, we'll use a simple JSON export
                    import json
                    import time
                    
                    if export_format == "JSON":
                        export_data = json.dumps({"qa_history": export_qa, "documents": export_docs}, indent=2)
                        filename = f"rag_session_export_{int(time.time())}.json"
                        mime_type = "file/json"
                    elif export_format == "CSV":
                        # Simple CSV export
                        export_data = "Question,Answer\n"
                        for qa in export_qa:
                            query = qa.get('query', '').replace('"', '""')
                            answer = qa.get('answer', '').replace('"', '""')
                            export_data += f'"{query}","{answer}"\n'
                        filename = f"rag_session_export_{int(time.time())}.csv"
                        mime_type = "file/csv"
                    else:  # HTML
                        # Simple HTML export
                        export_data = "<html><head><title>RAG Session Export</title></head><body>"
                        export_data += "<h1>Q&A History</h1>"
                        for qa in export_qa:
                            export_data += f"<div><h3>Q: {qa.get('query', '')}</h3><p>A: {qa.get('answer', '')}</p></div>"
                        export_data += "</body></html>"
                        filename = f"rag_session_export_{int(time.time())}.html"
                        mime_type = "file/html"
                    
                    # Create download link
                    download_link = get_download_link(export_data, filename, f"Download {export_format} Export")
                    st.markdown(download_link, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating export: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display export preview
            if qa_history or document_metadata:
                with st.expander("Export Preview"):
                    if qa_history and include_qa:
                        st.markdown("#### Q&A History Preview")
                        qa_df = pd.DataFrame([{
                            "Question": qa["query"],
                            "Answer": qa["answer"][:100] + "...",
                            "Confidence": qa["confidence_score"],
                            "Time": qa["timestamp"]
                        } for qa in qa_history[:5]])
                        st.dataframe(qa_df)
                    
                    if document_metadata and include_docs:
                        st.markdown("#### Document Metadata Preview")
                        # Show just a sample of the metadata
                        doc_keys = list(document_metadata.keys())[:5]
                        doc_preview = {k: document_metadata[k] for k in doc_keys}
                        st.json(doc_preview)

if __name__ == "__main__":
    main()
    
    # AI Code Review Tab
    if selected == "AI Code Review":
        st.markdown('<h2 class="sub-header">AI Code Review</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Code input and language selection
            code_input = st.text_area("Enter your code", height=300)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                language = st.selectbox("Select Language", ["Python", "JavaScript", "Java", "C++", "Ruby", "Go"])
            
            with col2:
                task_type = st.selectbox(
                    "Analysis Type",
                    ["Code Analysis", "Code Generation", "Code Optimization", "Bug Finding"]
                )
            
            if st.button("Analyze Code", type="primary"):
                if code_input.strip():
                    with st.spinner("Analyzing code..."):
                        try:
                            # Create request object
                            request = {
                                "code": code_input,
                                "task_type": task_type,
                                "language": language
                            }
                            
                            # Call code assistant API
                            response = requests.post(
                                "http://127.0.0.1:8001/code-assistant",
                                json=request
                            )
                            
                            if response.status_code == 200:
                                analysis_result = response.json()
                                
                                # Display results in an expandable section
                                with st.expander("Analysis Results", expanded=True):
                                    st.markdown("### Analysis Summary")
                                    st.markdown(analysis_result)
                                    
                                    # Add copy button for the analysis
                                    st.markdown(
                                        f"<button onclick=\"navigator.clipboard.writeText('{analysis_result}')\">Copy Analysis</button>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.error(f"Error analyzing code: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter some code to analyze.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display example code templates
            with st.expander("Example Code Templates"):
                st.markdown("### Example Code Templates")
                st.markdown("""
                Select a template to get started:
                
                **Python Example:**
                ```python
                def calculate_factorial(n):
                    if n < 0:
                        return None
                    if n == 0:
                        return 1
                    return n * calculate_factorial(n - 1)
                ```
                
                **JavaScript Example:**
                ```javascript
                function calculateFactorial(n) {
                    if (n < 0) return null;
                    if (n === 0) return 1;
                    return n * calculateFactorial(n - 1);
                }
                ```
                """)
            
            # Add tips and best practices
            with st.expander("Tips & Best Practices"):
                st.markdown("### Tips for Better Code Review")
                st.markdown("""
                1. **Clean Code**: Ensure your code is properly formatted before submission
                2. **Complete Functions**: Include complete function definitions
                3. **Context**: Add comments to explain complex logic
                4. **Scope**: Focus on specific functions or classes for better analysis
                5. **Language**: Make sure to select the correct programming language
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)