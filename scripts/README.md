# AlphaFold3 with ColabFold_Search MSA Pipeline

This guide explains how to integrate **ColabFold_Search** for multiple sequence alignment (MSA) generation with **AlphaFold3**. The ColabFold_Search tool efficiently searches for MSAs using protein sequences, and the resulting MSAs—with added taxid information—serve as input for AlphaFold3.

---

## Overview

This pipeline involves four main steps:

1. **Modify ColabFold_Search:** Update the code to perform a taxid-added search.
2. **Update AlphaFold3's MSA Extraction:** Adjust the species ID extraction function to handle ColabFold-specific data.
3. **Run the MSA Pipeline:** Process the input JSON to generate an MSA-enhanced JSON file.
4. **Run AlphaFold3 Inference:** Run AlphaFold3 inference using the processed JSON file.

---

## Prerequisites

- **Operating System:** Linux
- **Python Version:** 3.12
- **Package Manager:** [Mamba](https://github.com/mamba-org/mamba)
- **Memory:** Over 1000G is recommended for database pinning (optional)

---

## Step 1: Modify ColabFold_Search for Taxid Search

The original ColabFold_Search code pairs multimer MSAs and does not provide individual chain's taxid information. Update the code in `colabfold/mmseqs/search.py` with the provided version to preserve taxid data.

### Instructions:

1. **Clone the Repository and Replace Code:**

   ```bash
   git clone https://github.com/sokrypton/ColabFold.git /path/to/colabfold
   cp scripts/search.py /path/to/colabfold/mmseqs/search.py
   ```

2. **Install the Package from Source:**

   It is recommended to install ColabFold in an isolated environment:

   ```bash
   mamba create -n colabfold python=3.12
   mamba activate colabfold
   cd /path/to/colabfold
   pip install -e .
   ```

3. **Download and (Optionally) Pin the Database:**

   - **Download the ColabFold Database:**

     ```bash
     cd /path/to/colabfold
     ./setup_databases.sh /path/to/colabfold_db
     ```

   - **Pin the Database (if sufficient memory is available):**

     ```bash
     cd /path/to/colabfold_db
     sudo vmtouch -f -w -t -l -d -m 1000G *.idx
     ```

   - **Alternative (if insufficient memory):**

     ```bash
     cd /path/to/colabfold
     MMSEQS_NO_INDEX=1 ./setup_databases.sh /path/to/colabfold_db
     ```

---

## Step 2: Update the AlphaFold3 extract_species_ids Function

To correctly parse species IDs from the MSA features, update the `_UNIPROT_ENTRY_NAME_REGEX` in `src/alphafold3/data/msa_features.py`.

### Code Modification:

Replace the existing regex with the following:

```python
# UniProtKB SwissProt/TrEMBL dbs have the following description format:
# `db|UniqueIdentifier|EntryName`, e.g. `sp|P0C2L1|A3X1_LOXLA` or
# `tr|A0A146SKV9|A0A146SKV9_FUNHE`.
_UNIPROT_ENTRY_NAME_REGEX = re.compile(
    # UniProtKB TrEMBL or SwissProt database. Include 'cb' for ColabFold database.
    r'(?:cb|tr|sp)\|'
    # A primary accession number of the UniProtKB entry.
    r'(?:[A-Z0-9]{6,10})'
    # Occasionally there is an isoform suffix (e.g. _1 or _10) which we ignore.
    r'(?:_\d+)?\|'
    # TrEMBL: Same as AccessionId (6-10 characters).
    # SwissProt: A mnemonic protein identification code (1-5 characters).
    r'(?:[A-Z0-9]{1,10}_)'
    # A mnemonic species identification code.
    r'(?P<SpeciesId>[A-Z0-9]{1,5})'
)
```

---

## Step 3: Run the MSA Pipeline

1. **Prepare the Input JSON File:**

   Modify the template `scripts/af3_input/1N8Z.json` to include your specific protein data.

2. **Configure the `msa_config.yaml` File:**

   Update the paths in `msa_config.yaml` to point to the correct binaries and database directory:

   ```yaml
   mmseqs_bin_path: /path/to/colabfold/bin/mmseqs
   colabsearch_bin_path: /path/to/colabfold/bin/colabfold_search
   colabfold_db_dir: /path/to/colabfold_db
   ```

3. **Run the Pipeline:**

   Run the following command to process your input JSON and generate the MSA-added output:

   ```bash
   python scripts/mmseqs_pipeline.py \
       --input_json_path scripts/af3_input/1N8Z.json \
       --output_msa_dir scripts/colabfold_msas \
       --output_json_dir scripts/af3_input
   ```

   The processed JSON file will be saved as `scripts/af3_input/1N8Z_data.json`.

---

## Step 4: Run AlphaFold3 Inference

1. **Set Environment Variables:**

   Configure XLA flags and memory settings to optimize GPU usage:

   ```bash
   export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
   export XLA_PYTHON_CLIENT_PREALLOCATE=true
   export XLA_CLIENT_MEM_FRACTION=0.95
   ```

2. **Run AlphaFold3:**

   Run AlphaFold3 with the processed JSON file:

   ```bash
   python run_alphafold.py \
       --run_data_pipeline=false \  # Skip the original MSA search pipeline
       --json_path=scripts/af3_input/1N8Z_data.json \
       --model_dir=MODEL_DIR \    # Change this to your desired model path
       --output_dir=scripts/af3_output
   ```

## Acknowledgements

Thanks the [Protenix](https://github.com/bytedance/Protenix) and [Boltz-1](https://github.com/jwohlwend/boltz) team for providing the mmseqs2-based MSA searching pipeline.