from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
except ImportError as error:  # pragma: no cover - exercised when launching without optional UI dependency
    raise SystemExit("Streamlit n'est pas installé. Lancez `pip install -r requirements.txt`.") from error

from src.document_assistant.answer_generator import documentary_answer, llm_prompt
from src.document_assistant.chunker import chunk_pages
from src.document_assistant.comparator import compare_documents
from src.document_assistant.config import Settings
from src.document_assistant.document_loader import (
    DocumentLoadError, discover_documents, load_document,
)
from src.document_assistant.indexing import TfidfIndex, index_is_current, index_metadata
from src.document_assistant.retrieval import search_course
from src.document_assistant.llm_client import LLMConfigurationError, generate_answer
from src.document_assistant.summarizer import summarize_document


ROOT = Path(__file__).resolve().parent
SETTINGS = Settings.from_project_root(ROOT)


def build_document_index(use_ocr: bool = False) -> TfidfIndex:
    chunks = []
    source_paths = discover_documents(SETTINGS.pdf_dir)
    for path in source_paths:
        pages, _ = load_document(path, use_ocr=use_ocr)
        chunks.extend(chunk_pages(pages, SETTINGS.chunk_size, SETTINGS.chunk_overlap))
    index = TfidfIndex.build(chunks)
    index.save(SETTINGS.index_dir, source_paths)
    return index


@st.cache_data
def document_diagnostic(path_string: str, modified_ns: int, size: int, use_ocr: bool):
    del modified_ns, size
    _, diagnostic = load_document(Path(path_string), use_ocr=use_ocr)
    return diagnostic


@st.cache_resource
def load_index() -> TfidfIndex | None:
    try:
        return TfidfIndex.load(SETTINGS.index_dir)
    except (FileNotFoundError, ValueError, EOFError):
        return None


st.set_page_config(page_title="Document-Assistant", page_icon="📚", layout="wide")
st.title("Document-Assistant")
st.caption("Explorez le contenu de vos documents localement, quelle que soit leur extension.")

with st.sidebar:
    st.header("Documents")
    documents = discover_documents(SETTINGS.pdf_dir)
    st.write(f"{len(documents)} document(s) dans `data/raw/`")
    for pdf in documents:
        col_name, col_delete = st.columns([4, 1])
        col_name.write(pdf.name)
        if col_delete.button("×", key=f"delete-{pdf.name}", help="Supprimer ce document"):
            pdf.unlink()
            st.cache_resource.clear()
            st.rerun()
    if documents:
        with st.expander("État des documents"):
            for path in documents:
                try:
                    diagnostic = document_diagnostic(
                        str(path),
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                        use_ocr=False,
                    )
                except DocumentLoadError as error:
                    st.error(f"{path.name} : {error}")
                    continue
                st.markdown(f"**{path.name}**")
                st.caption(
                    f"{diagnostic.page_count} unité(s), "
                    f"{diagnostic.text_pages} avec texte, "
                    f"{diagnostic.character_count:,} caractères"
                )
                for warning in diagnostic.warnings:
                    st.warning(warning)
    uploaded = st.file_uploader("Ajouter un document")
    if uploaded:
        target = SETTINGS.pdf_dir / Path(uploaded.name).name
        if uploaded.size > SETTINGS.max_upload_mb * 1024 * 1024:
            st.error(f"Le fichier dépasse la limite de {SETTINGS.max_upload_mb} Mo.")
        elif target.exists():
            st.warning("Un fichier portant ce nom existe déjà.")
        else:
            SETTINGS.pdf_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getvalue())
            st.success("Document ajouté. Cliquez sur « Réindexer ».")
            st.cache_resource.clear()
    mode = st.radio("Mode de réponse", ["Documentaire local", "LLM optionnel"])
    use_ocr = st.checkbox("OCR optionnel pour les pages sans texte")
    if st.button("Réindexer", type="primary"):
        try:
            with st.spinner("Extraction et indexation..."):
                build_document_index(use_ocr=use_ocr)
            st.cache_resource.clear()
            st.success("Index reconstruit.")
            st.rerun()
        except (DocumentLoadError, ValueError) as error:
            st.error(str(error))

    index = load_index()
    if index:
        st.info(f"{len(index.chunks)} passages indexés")
        current = index_is_current(SETTINGS.index_dir, documents)
        if current is False:
            st.warning("L’index est obsolète : les documents ont changé. Réindexez.")
        elif current is None:
            st.caption("Index existant sans suivi de version des documents.")
        metadata = index_metadata(SETTINGS.index_dir)
        if metadata.get("created_at"):
            st.caption(f"Index créé le {metadata['created_at']}")
    result_count = st.slider("Nombre de résultats", 1, 10, SETTINGS.default_results)
    threshold = st.slider("Seuil de pertinence", 0.0, 1.0, SETTINGS.relevance_threshold, 0.01)

question = st.text_input("Votre question", placeholder="Ex. Quelles sont les hypothèses d'un test t ?")
st.subheader("Résumé local")
summary_documents = discover_documents(SETTINGS.pdf_dir)
if summary_documents:
    summary_path = st.selectbox(
        "Document à résumer",
        summary_documents,
        format_func=lambda path: path.name,
    )
    summary_count = st.slider("Nombre de phrases", 2, 10, 5, key="summary-count")
    if st.button("Générer le résumé"):
        try:
            with st.spinner("Analyse locale du document..."):
                summary, warnings = summarize_document(summary_path, summary_count, use_ocr)
            if summary:
                st.write(summary)
            else:
                st.warning("Aucun texte exploitable pour générer un résumé.")
            for warning in warnings:
                st.warning(warning)
        except DocumentLoadError as error:
            st.error(str(error))

st.subheader("Comparer deux documents")
if len(summary_documents) >= 2:
    first_document = st.selectbox(
        "Premier document",
        summary_documents,
        format_func=lambda path: path.name,
        key="comparison-first",
    )
    second_document = st.selectbox(
        "Deuxième document",
        summary_documents,
        index=1,
        format_func=lambda path: path.name,
        key="comparison-second",
    )
    comparison_count = st.slider(
        "Nombre de passages comparés", 1, 10, 5, key="comparison-count"
    )
    if st.button("Comparer les documents"):
        try:
            with st.spinner("Comparaison locale des documents..."):
                comparison = compare_documents(
                    first_document, second_document, comparison_count, use_ocr
                )
            if comparison.common_sentences:
                st.write("**Passages similaires**")
                for passage in comparison.common_sentences:
                    st.write(f"- {passage}")
            else:
                st.info("Aucun passage suffisamment similaire n'a été trouvé.")
            with st.container(border=True):
                st.write(f"**Éléments propres à {first_document.name}**")
                for passage in comparison.first_unique_sentences:
                    st.write(f"- {passage}")
            with st.container(border=True):
                st.write(f"**Éléments propres à {second_document.name}**")
                for passage in comparison.second_unique_sentences:
                    st.write(f"- {passage}")
            for warning in comparison.warnings:
                st.warning(warning)
        except (DocumentLoadError, ValueError) as error:
            st.error(str(error))
else:
    st.info("Ajoutez au moins deux documents pour lancer une comparaison.")

if st.button("Rechercher") and question.strip():
    index = load_index()
    if index is None:
        st.warning("Aucun index disponible. Réindexez d'abord les documents.")
    else:
        results = search_course(index, question, result_count, threshold)
        st.subheader("Résultat")
        if mode == "LLM optionnel":
            try:
                st.write(generate_answer(llm_prompt(question, results), question))
            except LLMConfigurationError as error:
                st.warning(str(error))
                st.write(documentary_answer(results))
        else:
            st.write(documentary_answer(results))
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(
            {"question": question, "answer": documentary_answer(results), "references": len(results)}
        )
        if results:
            st.subheader("Sources vérifiables")
            for result in results:
                with st.expander(f"{result.chunk.document_name} — page {result.chunk.page_number}"):
                    st.caption(
                        f"Score hybride : {result.score:.3f} "
                        f"(mots : {result.lexical_score:.3f}, caractères : {result.character_score:.3f})"
                    )
                    st.write(result.chunk.text)

if st.session_state.get("history"):
    with st.expander("Historique de la session"):
        for item in reversed(st.session_state.history):
            st.markdown(f"**Question :** {item['question']}")
            st.caption(f"{item['references']} passage(s) retrouvé(s)")
