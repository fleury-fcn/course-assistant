from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import streamlit as st
except ImportError as error:  # pragma: no cover - exercised when launching without optional UI dependency
    raise SystemExit("Streamlit n'est pas installé. Lancez `pip install -r requirements.txt`.") from error

from src.statistical_assistant.answer_generator import documentary_answer
from src.statistical_assistant.answer_generator import llm_prompt
from src.statistical_assistant.chunker import chunk_pages
from src.statistical_assistant.config import Settings
from src.statistical_assistant.document_loader import DocumentLoadError, discover_pdfs, load_pdf
from src.statistical_assistant.indexing import TfidfIndex
from src.statistical_assistant.retrieval import search_course
from src.statistical_assistant.llm_client import LLMConfigurationError, generate_answer


ROOT = Path(__file__).resolve().parent
SETTINGS = Settings.from_project_root(ROOT)


def build_index(use_ocr: bool = False) -> TfidfIndex:
    chunks = []
    for path in discover_pdfs(SETTINGS.pdf_dir):
        pages, _ = load_pdf(path, use_ocr=use_ocr)
        chunks.extend(chunk_pages(pages, SETTINGS.chunk_size, SETTINGS.chunk_overlap))
    index = TfidfIndex.build(chunks)
    index.save(SETTINGS.index_dir)
    return index


@st.cache_resource
def load_index() -> TfidfIndex | None:
    try:
        return TfidfIndex.load(SETTINGS.index_dir)
    except (FileNotFoundError, ValueError, EOFError):
        return None


st.set_page_config(page_title="Assistant de statistique", page_icon="📚", layout="wide")
st.title("Assistant de statistique inférentielle")
st.caption("Recherche documentaire locale dans vos cours PDF — TF-IDF, sans clé API.")

with st.sidebar:
    st.header("Documents")
    pdfs = discover_pdfs(SETTINGS.pdf_dir)
    st.write(f"{len(pdfs)} document(s) dans `data/raw/`")
    for pdf in pdfs:
        col_name, col_delete = st.columns([4, 1])
        col_name.write(pdf.name)
        if col_delete.button("×", key=f"delete-{pdf.name}", help="Supprimer ce document"):
            pdf.unlink()
            st.cache_resource.clear()
            st.rerun()
    uploaded = st.file_uploader("Ajouter un PDF", type=["pdf"])
    if uploaded:
        target = SETTINGS.pdf_dir / Path(uploaded.name).name
        if uploaded.size > SETTINGS.max_upload_mb * 1024 * 1024:
            st.error(f"Le fichier dépasse la limite de {SETTINGS.max_upload_mb} Mo.")
        elif target.exists():
            st.warning("Un fichier portant ce nom existe déjà.")
        else:
            SETTINGS.pdf_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getvalue())
            st.success("PDF ajouté. Cliquez sur « Réindexer ».")
            st.cache_resource.clear()
    mode = st.radio("Mode de réponse", ["Documentaire local", "LLM optionnel"])
    use_ocr = st.checkbox("OCR optionnel pour les pages sans texte")
    if st.button("Réindexer", type="primary"):
        try:
            with st.spinner("Extraction et indexation..."):
                build_index(use_ocr=use_ocr)
            st.cache_resource.clear()
            st.success("Index reconstruit.")
            st.rerun()
        except (DocumentLoadError, ValueError) as error:
            st.error(str(error))

    index = load_index()
    if index:
        st.info(f"{len(index.chunks)} passages indexés")
    result_count = st.slider("Nombre de résultats", 1, 10, SETTINGS.default_results)
    threshold = st.slider("Seuil de pertinence", 0.0, 1.0, SETTINGS.relevance_threshold, 0.01)

question = st.text_input("Votre question", placeholder="Ex. Quelles sont les hypothèses d'un test t ?")
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
