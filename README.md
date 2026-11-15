# Intelligent Phishing Campaign Behaviour Detection — Nazario Dataset

This Streamlit application is adapted to the uploaded `Nazario.csv` dataset.

## Dataset schema
- `sender`
- `receiver`
- `date`
- `subject`
- `body`
- `urls`
- `label`

The app loads `data/Nazario.csv` automatically. You can also upload another compatible CSV in the sidebar.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model workflow
1. Parse email dates and extract sender domains.
2. Derive transparent keyword/URL features from subject and body.
3. Convert each email into a categorical observation symbol.
4. Group and sort emails chronologically using sender domain, recipient, or one mailbox sequence.
5. Train a five-state discrete HMM using Baum-Welch.
6. Use scaled Forward/Backward for sequence likelihood and posterior state probabilities.
7. Use Viterbi to decode the most likely latent-state path.
8. Visualize transition/emission probabilities and sequence-level investigation details.

## Important dataset limitation
The supplied CSV contains 1,565 records and all observed `label` values are `1`. It therefore does not contain both phishing and legitimate classes. Do not report binary classification accuracy, precision, recall, F1, or ROC-AUC from this file. To evaluate binary detection, add an authorized and documented legitimate-email dataset with compatible fields. To evaluate decoded campaign stages, obtain manually annotated stage labels.

The stage names are heuristic interpretations of unsupervised hidden states. They are not verified ground truth. Sender-domain grouping is an approximation and does not prove that messages from the same domain belong to one campaign.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Add `data/Nazario.csv` only if you have permission to redistribute it and accept that anyone with repository access may be able to access the raw email content.
3. For a public repository, do not commit unredacted sender/receiver addresses or private email bodies. Instead, omit the CSV and upload it through the app's file uploader when needed, or prepare a properly anonymized dataset.
4. In Streamlit Community Cloud, connect the repository, select the branch, set main file path to `app.py`, and deploy.

## Privacy
The bundled dataset may contain personal email addresses and message content. Keep the repository private unless the dataset's license and privacy status clearly permit public redistribution. Use only data you are authorized to process.
