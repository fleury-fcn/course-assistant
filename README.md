# Document-Assistant

Document-Assistant est une application locale en français pour explorer le
contenu de documents, quel que soit leur format courant. Elle extrait les
pages ou le texte, indexe les passages avec une recherche hybride
TF-IDF mots/caractères et affiche les sources vérifiables. Elle fonctionne sans
clé API et ne transmet pas les documents à un service distant par défaut.

## Installation et lancement

```bash
cd ~/AI-Research/projects/statistical-assistant
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.inspect_documents
python -m scripts.build_document_index
streamlit run app.py
```

Les dépendances déjà présentes dans le venv peuvent être conservées. Les tests
de développement s'installent avec
`pip install -r requirements-dev.txt`, puis se lancent par `pytest -q`.

## Architecture

- `document_loader.py` découvre et extrait les formats pris en charge.
- `text_cleaner.py` fournit un nettoyage prudent, sans remplacer le texte brut.
- `chunker.py` crée des passages par page avec chevauchement configurable.
- `indexing.py` construit et sauvegarde l'index TF-IDF local.
- Les métadonnées de l'index suivent les empreintes des documents pour signaler
  un index obsolète après une modification ou un nouvel import.
- `retrieval.py` classe les passages par similarité cosinus.
- `citations.py` construit les références à partir des métadonnées, jamais à
  partir d'un texte généré.
- `answer_generator.py` fournit le mode documentaire et le contexte contrôlé
  destiné à une future intégration LLM.
- `summarizer.py` produit des résumés extractifs locaux sans API distante.
- `comparator.py` compare deux documents avec une similarité TF-IDF locale.
- `app.py` fournit l'interface Streamlit en français.

## Utilisation

Déposez vos documents dans `data/raw/`, ou ajoutez-les avec l'interface.
Les extensions prises en charge sont :

- PDF, XPS, EPUB, MOBI, CBZ, FB2 et SVG via PyMuPDF ;
- TXT, Markdown, CSV, TSV, JSON, XML, HTML, YAML et fichiers de code ;
- DOCX, PPTX et XLSX via leur contenu XML interne.

Cliquez sur **Réindexer**, saisissez une question puis consultez les passages
et leurs pages ou unités documentaires. Un score de recherche est un indice de proximité,
pas une probabilité de vérité. Une question sans vocabulaire commun renvoie
explicitement zéro résultat.

La barre latérale contient aussi un état détaillé des documents : nombre
d'unités, unités contenant du texte, volume de caractères extraits et
avertissements éventuels.

La section **Résumé local** permet de sélectionner un document et de générer
un résumé extractif de 2 à 10 phrases. La section **Comparer deux documents**
identifie les passages similaires et les passages les plus spécifiques à chaque
document. Ces deux fonctions sont exécutées localement et leurs résultats
doivent être vérifiés comme des aides à l'analyse, pas comme des conclusions
automatiques.

L'index sérialise un objet Python : ne chargez jamais `data/index/tfidf.pkl`
provenant d'une source non fiable. Les index sont ignorés par Git et peuvent
être supprimés puis reconstruits avec `python -m scripts.build_document_index`.

## Limites et prochaines étapes

La recherche utilise un score hybride mots/caractères : elle tolère mieux les
variantes lexicales, mais ne remplace pas des embeddings. L'OCR est disponible
si `pytesseract`, Pillow et le binaire Tesseract sont installés
explicitement (`brew install tesseract`, puis `pip install pytesseract Pillow`).
Le LLM est optionnel et compatible avec une API OpenAI-compatible : définir
`LLM_API_KEY`, `LLM_MODEL` et éventuellement `LLM_BASE_URL` dans l'environnement.
Les extraits sélectionnés sont alors transmis au fournisseur ; sans ces variables,
l'application revient au mode documentaire local. Les clés ne sont jamais
enregistrées ni journalisées.

## Renommer le dépôt GitHub

Le code et l'interface utilisent désormais **Document-Assistant**. Pour
renommer aussi le dépôt distant, ouvrir **Settings > General > Repository name**
sur GitHub et choisir `Document-Assistant`, puis mettre à jour le remote local :

```bash
git remote set-url origin git@github.com:fleury-fcn/Document-Assistant.git
```
