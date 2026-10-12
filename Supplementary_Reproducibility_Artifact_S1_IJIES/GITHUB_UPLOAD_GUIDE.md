# GitHub Upload Guide - Artifact S1

## Recommended repository name
`switchgear-ultrasound-audit-20266581`

## Upload these files
1. Create a **public** GitHub repository.
2. Extract `Supplementary_Reproducibility_Artifact_S1_IJIES_EXPANDED_FINAL.zip` locally.
3. Upload the **extracted contents** to the repository root. Do not upload only the ZIP as the repository contents.
4. Keep `README.md` visible at the root.
5. Commit the scripts, CSV outputs, `config.json`, `requirements.txt`, `timing_protocol.md`, `ARTIFACT_MANIFEST.md`, and `CITATION.cff`.

## Do NOT upload unless redistribution permission is confirmed
- source `.mat` files
- raw audio
- private/unreleased notebooks containing redistributable data
- credentials, tokens, or local paths

The included `.gitignore` is intended to reduce accidental data uploads, but review the staged files before every commit.

## Recommended release snapshot
After the repository is complete:
1. Create a GitHub Release with tag `review2-s1-v1.0`.
2. Attach the exact journal supplementary ZIP: `Supplementary_Reproducibility_Artifact_S1_IJIES_EXPANDED_FINAL.zip`.
3. Copy the repository URL.
4. Insert that URL into the manuscript's **Code and Reproducibility Artifact** paragraph and the response letter before final submission if the journal permits/requests a public link.

## Journal submission
Upload the 18-page revised manuscript and the response letter in their designated fields. If the IJIES portal provides **Supplementary / Additional / Other file**, upload the exact S1 ZIP there. If no such field is available, keep the public GitHub link in the manuscript/response and send the ZIP to the editorial office only if requested or permitted.

## Terminology note
The repository documentation should refer to the two branches as **Time Domain** and **Frequency Domain**. Use “stored one-sided FFT spectrum” only when describing the format of the supplied frequency-domain arrays.
