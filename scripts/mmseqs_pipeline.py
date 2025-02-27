# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

import argparse
import os
from os.path import join as opjoin
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from glob import glob
from tqdm import tqdm
from functools import partial
import tempfile
import json
from copy import deepcopy
from pathlib import Path
import yaml
import warnings
import shutil

@dataclass
class LocalColabFoldConfig:
    """Configuration for ColabFold search."""

    colabsearch: str
    query_fpath: str
    db_dir: str
    results_dir: str
    mmseqs_path: Optional[str] = None
    db1: str = "uniref30_2302_db"
    db2: Optional[str] = None
    db3: Optional[str] = "colabfold_envdb_202108_db"
    use_env: int = 1
    filter: int = 1
    db_load_mode: int = 0
    add_toxid: int = 0
    threads: int = 32

def run_colabfold_search(config: LocalColabFoldConfig) -> str:
    """Run ColabFold search with given configuration."""
    cmd = [config.colabsearch, config.query_fpath, config.db_dir, config.results_dir]

    # Add optional parameters
    if config.db1:
        cmd.extend(["--db1", config.db1])
    if config.db2:
        cmd.extend(["--db2", config.db2])
    if config.db3:
        cmd.extend(["--db3", config.db3])
    if config.mmseqs_path:
        cmd.extend(["--mmseqs", config.mmseqs_path])
    else:
        cmd.extend(["--mmseqs", "mmseqs"])
    if config.use_env:
        cmd.extend(["--use-env", str(config.use_env)])
    if config.filter:
        cmd.extend(["--filter", str(config.filter)])
    if config.db_load_mode:
        cmd.extend(["--db-load-mode", str(config.db_load_mode)])
    if config.add_toxid:
        cmd.extend(["--add-toxid", str(config.add_toxid)])
    if config.threads:
        cmd.extend(["--threads", str(config.threads)])

    cmd = " ".join(cmd)
    os.system(cmd)


def write_log(
    msg: str,
    fname: str,
    log_root: str,
) -> None:
    basename = fname.split(".")[0]
    with open(opjoin(log_root, f"{basename}-{msg}"), "w") as f:
        pass

def read_a3m(a3m_file: str) -> tuple[List[str], List[str]]:
    """read a3m file from output of mmseqs

    Args:
        a3m_file (str): the a3m file searched by mmseqs(colabfold search)

    Returns:
        tuple[List[str], List[str]]: the header and seqs of a3m files
    """
    heads = []
    seqs = []
    # Record the row index. The index before this index is the MSA of Uniref30 DB,
    # and the index after this index is the MSA of ColabfoldDB.
    uniref_index = 0
    with open(a3m_file, "r") as infile:
        for idx, line in enumerate(infile):
            if line.startswith(">"):
                heads.append(line)
                if idx == 0:
                    query_name = line
                elif idx > 0 and line == query_name:
                    uniref_index = idx
            else:
                seqs.append(line)
    return heads, seqs, uniref_index


def read_m8(m8_file: str) -> Dict[str, str]:
    """the uniref_tax.m8 from output of mmseqs

    Args:
        m8_file (str): the uniref_tax.m8 from output of mmseqs(colabfold search)

    Returns:
        Dict[str, str]: the dict mapping uniref hit_name to NCBI TaxID
    """
    uniref_to_ncbi_taxid = {}
    with open(m8_file, "r") as infile:
        for line in infile:
            line_list = line.replace("\n", "").split("\t")
            hit_name = line_list[1]
            ncbi_taxid = line_list[2]
            uniref_to_ncbi_taxid[hit_name] = ncbi_taxid
    return uniref_to_ncbi_taxid


def update_a3m(
    a3m_path: str,
    uniref_to_ncbi_taxid: Dict,
    save_root: str,
) -> None:
    """add NCBI TaxID to header if "UniRef" in header

    Args:
        a3m_path (str): the original a3m path returned by mmseqs(colabfold search)
        uniref_to_ncbi_taxid (Dict): the dict mapping uniref hit_name to NCBI TaxID
        save_root (str): the updated a3m
    """
    heads, seqs, uniref_index = read_a3m(a3m_path)
    fname = a3m_path.split("/")[-1]
    out_a3m_path = opjoin(save_root, fname)
    with open(out_a3m_path, "w") as ofile:
        for idx, (head, seq) in enumerate(zip(heads, seqs)):
            uniref_id = head.split("\t")[0][1:]
            ncbi_taxid = uniref_to_ncbi_taxid.get(uniref_id, None)
            if (ncbi_taxid is not None) and (idx < (uniref_index // 2)):
                if not uniref_id.startswith("UniRef100_"):
                    head = head.replace(
                        uniref_id, f"UniRef100_{uniref_id}_{ncbi_taxid}/"
                    )
                else:
                    head = head.replace(uniref_id, f"{uniref_id}_{ncbi_taxid}/")
            ofile.write(f"{head}{seq}")


def process_one_file(
    fname: str, msa_root: str, save_fname: str, logger: Callable
) -> None:
    with open(file_path := opjoin(msa_root, fname), "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                pdb_line = line
            if i == 1:
                if len(line) == 1:
                    logger("empty_query_seq", fname)
                    return
                query_line = line
                break

    os.makedirs(sub_dir_path := opjoin(msa_root, f"{save_fname}"), exist_ok=True)
    

    with open(file_path, "r") as f:
        lines = f.readlines()
    
    origin_query_seq = lines[1].strip()
    uniref100_lines = [">query\n", f"{origin_query_seq}\n"]
    other_lines = [">query\n", f"{origin_query_seq}\n"]

    for i, line in enumerate(lines[2:]):
        if i % 2 == 0:
            # header
            if not line.startswith(">"):
                logger(f"bad_header_{i}", fname)
                return
            seq = lines[i + 1]

            if line.startswith(">UniRef100"):
                uniref_id = line.split("\t")[0].split("_")
                if len(uniref_id) == 3:
                    header = f">cb|{uniref_id[1]}|{uniref_id[1]}_{uniref_id[2]}\n"
                else:
                    header = f">cb|{uniref_id[1]}|{uniref_id[1]}/\n"
                uniref100_lines.extend([header, seq])
            else:
                other_lines.extend([line, seq])

    assert len(other_lines) + len(uniref100_lines) - 2 == len(lines)

    other_lines = other_lines[0:2] + other_lines[4:]
    for i, line in enumerate(other_lines):
        if i > 0 and i % 2 == 0:
            assert "\t" in line
    pairing_exist, non_pairing_exist = len(uniref100_lines) > 2, len(other_lines) > 2
    if pairing_exist:
        with open(opjoin(sub_dir_path, "uniref100_hits.a3m"), "w") as f:
            for line in uniref100_lines:
                f.write(line)
    if non_pairing_exist:
        with open(opjoin(sub_dir_path, "mmseqs_other_hits.a3m"), "w") as f:
            for line in other_lines:
                f.write(line)
    assert pairing_exist or non_pairing_exist, f"No pairing or non_pairing a3m for {fname}"
    
    return origin_query_seq


def write_to_af3_template(
    fold_input_fpath: str,
    msa_dir: str,
    out_dir: str,
    unpairing_db: List[str] = ["uniref100", "mmseqs_other"],
    pairing_db: List[str] = ["uniref100"],
):
    
    fold_input_fpath = Path(fold_input_fpath)
    msa_dir = Path(msa_dir)
    out_dir = Path(out_dir)
    with open(msa_dir / "msa_chain_seq.json", "r") as f:
        msa_chain_seq = json.load(f)
    msa_seq_chain = {v: k for k, v in msa_chain_seq.items()}

    with open(fold_input_fpath, "r") as f:
        fold_input = json.load(f)
    
    # copy fold_input
    fold_input_out = deepcopy(fold_input)

    for i, chain in enumerate(fold_input["sequences"]):
        chain_seq = chain["protein"]["sequence"]
        chain_id_msa = msa_seq_chain[chain_seq]
        non_pairing_a3m = ""
        for db in unpairing_db:
            non_pairing_a3m_fpath = msa_dir / f"{chain_id_msa}/{db}_hits.a3m"
            if not non_pairing_a3m_fpath.exists():
                continue
            with open(non_pairing_a3m_fpath, "r") as f:
                non_pairing_a3m += f.read()
        non_pairing_a3m = non_pairing_a3m.replace("\t", " ")
        fold_input_out["sequences"][i]["protein"]["unpairedMsa"] = non_pairing_a3m
        fold_input_out["sequences"][i]["protein"]["templates"] = []
        
        pairing_a3m = ""
        for db in pairing_db:
            pairing_a3m_fpath = msa_dir / f"{chain_id_msa}/{db}_hits.a3m"
            if not pairing_a3m_fpath.exists():
                continue
            with open(pairing_a3m_fpath, "r") as f:
                pairing_a3m += f.read()
        pairing_a3m = pairing_a3m.replace("\t", " ")
        fold_input_out["sequences"][i]["protein"]["pairedMsa"] = pairing_a3m
        fold_input_out["sequences"][i]["protein"]["templates"] = []

    with open(out_dir / f"{fold_input_out['name']}_data.json", "w") as f:
        f.write(json.dumps(fold_input_out, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_json_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--msa_config_path",
        type=str,
        default="scripts/msa_config.yaml",
    )
    parser.add_argument(
        "--output_msa_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--add_toxid",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    
    # Step 1 input json file to fasta file
    with open(args.input_json_path, "r") as f:
        fold_input = json.load(f)
    os.makedirs(args.output_msa_dir, exist_ok=True)
    tmp_fasta_fpath = opjoin(args.output_msa_dir, f"{fold_input['name']}.fasta")
    
    num_prot_chains = 0
    with open(tmp_fasta_fpath, "w") as f:
        for chain in fold_input["sequences"]:
            if "protein" not in chain: continue
            f.write(f">{fold_input['name']}_{chain['protein']['id']}\n{chain['protein']['sequence']}\n")
            num_prot_chains += 1
    if num_prot_chains == 0:
        raise ValueError(f"No protein chains found in {args.input_json_path}")
    if num_prot_chains == 1 and args.add_toxid == 1:
        warnings.warn(f"Monomer found in {args.input_json_path}, we recommend to set add_toxid to 0")
    if num_prot_chains > 1 and args.add_toxid == 0:
        warnings.warn(f"Multimer found in {args.input_json_path}, we change add_toxid to 1")
        args.add_toxid = 1
    
    # Step 2 run colabfold search
    print(f"Running colabfold search for {fold_input['name']}...")
    with open(args.msa_config_path, "r") as f:
        msa_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    colab_config = LocalColabFoldConfig(
        colabsearch=msa_config.pop("colabsearch_bin_path"),
        query_fpath=tmp_fasta_fpath,
        db_dir=msa_config.pop("colabfold_db_dir"),
        mmseqs_path=msa_config.pop("mmseqs_bin_path"),
        results_dir=args.output_msa_dir,
        add_toxid=args.add_toxid,
        **msa_config
    )
    run_colabfold_search(colab_config)

    # Step 3 update a3m with NCBI TaxID
    save_root = None
    if args.add_toxid == 1:
        a3m_paths = glob(f"{args.output_msa_dir}/*.a3m")
        m8_file = f"{args.output_msa_dir}/uniref_tax.m8"
        uniref_to_ncbi_taxid = read_m8(m8_file)
        save_root = tempfile.mkdtemp()
        for a3m_path in tqdm(a3m_paths):
            update_a3m(
                a3m_path=a3m_path,
                uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
                save_root=save_root,
            )

    logger = partial(write_log, log_root=args.output_msa_dir)
    msa_chain_seq = {}
    save_root = args.output_msa_dir if save_root is None else save_root
    for i, fname in enumerate(glob(f"{save_root}/*.a3m")):
        query = process_one_file(
            fname=fname,
            msa_root=args.output_msa_dir,
            save_fname=str(i),
            logger=logger,
        )
        msa_chain_seq[str(i)] = query
    with open(f"{args.output_msa_dir}/msa_chain_seq.json", "w") as f:
        f.write(json.dumps(msa_chain_seq, indent=4))
    if args.add_toxid == 1:
        shutil.rmtree(save_root)
    
    # Step 4 write to alphafold3 input json template
    os.makedirs(args.output_json_dir, exist_ok=True)
    write_to_af3_template(
        fold_input_fpath=args.input_json_path,
        msa_dir=args.output_msa_dir,
        out_dir=args.output_json_dir,
    )
