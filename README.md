# Statistical Learning Assistant

Assistant local en français pour rechercher des passages dans des cours PDF de
statistique inférentielle. La première version utilise PyMuPDF pour l'extraction
et TF-IDF/scikit-learn pour une recherche lexicale. Elle fonctionne sans clé API
et ne transmet pas les PDF à un service distant.

## Installation et lancement

```bash
cd ~/AI-Research/projects/statistical-assistant
source .venv/bin/activate
pip install -r requirements.txt
python scripts/inspect_pdf.py
python scripts/build_index.py
streamlit run app.py
```

Les dépendances déjà présentes dans le venv peuvent être conservées ; installez
Streamlit seulement s'il manque. Les tests de développement s'installent avec
`pip install -r requirements-dev.txt`, puis se lancent par `pytest -q`.

## Architecture

- `document_loader.py` extrait les pages physiques et produit des diagnostics.
- `text_cleaner.py` fournit un nettoyage prudent, sans remplacer le texte brut.
- `chunker.py` crée des passages par page avec chevauchement configurable.
- `indexing.py` construit et sauvegarde l'index TF-IDF local.
- `retrieval.py` classe les passages par similarité cosinus.
- `citations.py` construit les références à partir des métadonnées, jamais à
  partir d'un texte généré.
- `answer_generator.py` fournit le mode documentaire et le contexte contrôlé
  destiné à une future intégration LLM.
- `app.py` fournit l'interface Streamlit en français.

## Utilisation

Déposez un ou plusieurs PDF dans `data/raw/`, ou ajoutez-les avec l'interface.
Cliquez sur **Réindexer**, saisissez une question puis consultez les passages
et leurs pages physiques. Un score TF-IDF est un indice de proximité lexicale,
pas une probabilité de vérité. Une question sans vocabulaire commun renvoie
explicitement zéro résultat.

L'index sérialise un objet Python : ne chargez jamais `data/index/tfidf.pkl`
provenant d'une source non fiable. Les index sont ignorés par Git et peuvent
être supprimés puis reconstruits avec `python scripts/build_index.py`.

## Limites et prochaines étapes

La recherche utilise maintenant un score hybride mots/caractères : elle tolère
les variantes lexicales, mais ne remplace pas des embeddings. L'OCR est
disponible si `pytesseract`, Pillow et le binaire Tesseract sont installés
explicitement (`brew install tesseract`, puis `pip install pytesseract Pillow`).
Le LLM est optionnel et compatible avec une API OpenAI-compatible : définir
`LLM_API_KEY`, `LLM_MODEL` et éventuellement `LLM_BASE_URL` dans l'environnement.
Les extraits sélectionnés sont alors transmis au fournisseur ; sans ces variables,
l'application revient au mode documentaire local. Les clés ne sont jamais
enregistrées ni journalisées.
