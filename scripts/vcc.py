"""Adapt VCC data to the original GEARS prepare/train workflow and export H5AD."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# Running this file always uses this checkout, including with an installed cell-gears.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LINGSHU = REPO.parent / 'Lingshu-Cell'
CONTROLS = {'ctrl', 'control', 'non-targeting', 'non-perturbation', 'nt'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_mapping(path):
    table = pd.read_csv(path, sep='\t', header=None, names=['id', 'symbol'], dtype=str)
    if table.isna().any().any():
        raise ValueError('Gene mapping must contain two tab-separated columns: Ensembl ID, symbol')
    return dict(zip(table.id, table.symbol))


def symbol(value, mapping, allow_unmapped=False):
    if pd.isna(value) or not str(value).strip():
        raise ValueError('Missing gene/perturbation label')
    value = str(value).strip()
    key = value.split('.')[0] if value.startswith('ENSG') else value
    result = mapping.get(key, key)
    if result.startswith('ENSG') and not allow_unmapped:
        raise ValueError(f'Unmapped Ensembl gene: {value}')
    return result


def condition(value, mapping):
    if pd.isna(value):
        raise ValueError('Missing perturbation label')
    parts = [p.strip() for p in str(value).split('+')]
    if any(not p for p in parts):
        raise ValueError(f'Empty perturbation label: {value!r}')
    genes = [symbol(p, mapping) for p in parts if p.casefold() not in CONTROLS]
    if not genes:
        return 'ctrl'
    if len(genes) != 1:
        raise ValueError(f'This VCC adapter expects single-gene perturbations: {value}')
    return genes[0] + '+ctrl'


def read_panel(path, mapping):
    path = Path(path)
    if path.suffix.lower() == '.json':
        values = json.loads(path.read_text())['target_gene_to_token_id_info']['gene_names_ordered']
    else:
        table = pd.read_csv(path, header=None, dtype=str)
        if table.shape[1] != 1:
            raise ValueError('Gene panel must be a one-column CSV/text file or Lingshu vocabulary JSON')
        values = table.iloc[:, 0].tolist()
        if values and str(values[0]).casefold() in {'gene', 'genes', 'gene_name', 'gene_symbol', 'symbol'}:
            values = values[1:]
    genes = [symbol(value, mapping) for value in values]
    if len(genes) < 20 or len(set(g.upper() for g in genes)) != len(genes):
        raise ValueError('Gene panel must contain at least 20 unique gene symbols (GEARS DE20)')
    return genes


def read_conditions(path, mapping, cell_type):
    table = pd.read_csv(path)
    if not {'target_gene', 'n_cells'}.issubset(table):
        raise ValueError('Conditions CSV must contain target_gene and n_cells')
    labels = table.target_gene.map(lambda x: condition(x, mapping))
    table = table.loc[labels != 'ctrl'].copy()
    table['target_gene'] = labels.loc[table.index].str.removesuffix('+ctrl')
    counts = pd.to_numeric(table.n_cells, errors='raise')
    if (table.empty or table.target_gene.duplicated().any() or
            not np.isfinite(counts).all() or (counts <= 0).any() or (counts % 1 != 0).any()):
        raise ValueError('Conditions require unique perturbations and positive integer n_cells')
    table['n_cells'] = counts.astype('int64')
    if 'cell_type' in table and not table.cell_type.eq(cell_type).all():
        raise ValueError(f'Conditions must contain only cell_type={cell_type}')
    table['cell_type'] = cell_type
    return table[['cell_type', 'target_gene', 'n_cells']].reset_index(drop=True)


def load_adata(path, panel, mapping, *, split, target_key, cell_type, layer, input_scale):
    source = ad.read_h5ad(path)
    if target_key not in source.obs:
        raise ValueError(f'{path}: missing obs[{target_key!r}]')
    if 'cell_type' in source.obs and not source.obs.cell_type.eq(cell_type).all():
        raise ValueError(f'{path}: expected a single cell_type={cell_type}')
    name_key = next((key for key in ['gene_name', 'gene_symbols', 'symbol', 'feature_name']
                     if key in source.var), None)
    names = source.var[name_key] if name_key else source.var_names
    names = [symbol(value, mapping, allow_unmapped=True) for value in names]
    lookup = {name: i for i, name in enumerate(names)}
    missing = sorted(set(panel) - set(lookup))
    duplicates = pd.Index(names)[pd.Index(names).duplicated()].intersection(panel).tolist()
    if missing or duplicates:
        raise ValueError(f'{path}: missing panel genes={missing[:20]}, duplicate symbols={duplicates[:20]}')
    labels = source.obs[target_key].map(lambda x: condition(x, mapping)).to_numpy()
    # Only training controls may provide basal expression during training/inference.
    rows = np.flatnonzero(labels != 'ctrl') if split == 'val' else np.arange(source.n_obs)
    matrix = source.layers[layer] if layer else source.X
    matrix = sparse.csr_matrix(matrix[rows][:, [lookup[g] for g in panel]], dtype=np.float32)
    matrix.sum_duplicates()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f'{path}: expression must be finite and nonnegative')
    if input_scale == 'counts':
        if not np.allclose(matrix.data, np.rint(matrix.data), rtol=0, atol=1e-5):
            raise ValueError(f'{path}: noninteger counts; use --input-scale lognorm for normalized X')
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        if (totals <= 0).any():
            raise ValueError(f'{path}: cells with zero counts on the selected gene panel')
        matrix = matrix.multiply((10000 / totals)[:, None]).tocsr()
        np.log1p(matrix.data, out=matrix.data)
    obs = pd.DataFrame({'condition': labels[rows], 'cell_type': cell_type, 'vcc_split': split},
                       index=[f'{split}_{name}' for name in source.obs_names[rows]])
    if not obs.index.is_unique:
        raise ValueError(f'{path}: cell identifiers must be unique')
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame({'gene_name': panel}, index=panel))


def fixed_split(train, val, prediction_targets):
    train = set(train) - {'ctrl'}
    val = set(val) - {'ctrl'}
    test = {gene + '+ctrl' for gene in prediction_targets}
    if not train or not val or not test or train & val or (train | val) & test:
        raise ValueError('Train/validation/test perturbations must be nonempty and disjoint')
    return {'train': ['ctrl'] + sorted(train), 'val': sorted(val), 'test': []}


def check_coverage(targets, available, source='GO graph'):
    missing = sorted(set(targets) - set(available))
    if missing:
        raise ValueError(f'Perturbation targets missing from the {source} (none were dropped): {missing}')


def prepare(args):
    from gears import PertData

    work = args.work_dir.resolve()
    if work.name != work.name.lower() or (work.exists() and any(work.iterdir())):
        raise ValueError('--work-dir must have a lowercase name and be empty for preparation')
    mapping = read_mapping(args.gene_map)
    panel = read_panel(args.gene_panel, mapping)
    conditions = read_conditions(args.conditions or args.data_root / 'test/pert_counts_Test.csv',
                                 mapping, args.cell_type)
    common = dict(panel=panel, mapping=mapping, target_key=args.target_key,
                  cell_type=args.cell_type, layer=args.layer, input_scale=args.input_scale)
    train = load_adata(args.train_h5ad or args.data_root / 'train/adata_Training.h5ad',
                       split='train', **common)
    val = load_adata(args.val_h5ad or args.data_root / 'validation/adata_Validation.h5ad',
                     split='val', **common)
    if not train.obs.condition.eq('ctrl').any():
        raise ValueError('Training H5AD must include non-targeting control cells')
    splits = fixed_split(train.obs.condition, val.obs.condition, conditions.target_gene)
    targets = sorted({c.removesuffix('+ctrl') for c in splits['train'] + splits['val'] if c != 'ctrl'}
                     | set(conditions.target_gene))
    check_coverage(targets, panel, source='gene panel')
    work.mkdir(parents=True, exist_ok=True)
    # GEARS supports a custom perturbation graph: use VCC target identities only.
    # Held-out target names are public metadata; no test expression is read.
    with (work / 'pert_genes.pkl').open('wb') as handle:
        pickle.dump(targets, handle)
    data = PertData(str(work.parent), gene_set_path=str(work / 'pert_genes.pkl'))
    check_coverage(targets, data.gene2go)
    combined = ad.concat([train, val], merge='same')
    combined.uns['evaluation'] = {'expression_scale': 'lognorm', 'producer': 'GEARS VCC'}
    np.random.seed(args.seed)
    data.new_data_process(dataset_name=work.name, adata=combined)
    check_coverage(targets, data.pert_names)
    with (work / 'split.pkl').open('wb') as handle:
        pickle.dump(splits, handle)
    train[train.obs.condition == 'ctrl'].copy().write_h5ad(work / 'control.h5ad')
    conditions.to_csv(work / 'conditions.csv', index=False)
    (work / 'panel.txt').write_text('\n'.join(panel) + '\n')
    write_json(work / 'vcc.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'genes': panel, 'pert_genes': data.pert_names.tolist(), 'cell_type': args.cell_type,
        'gene_panel_sha256': hashlib.sha256('\n'.join(panel).encode()).hexdigest(),
        'expression_scale': 'lognorm', 'input_scale': args.input_scale,
        'normalize_total': 10000 if args.input_scale == 'counts' else 'provided by user',
        'n_train': train.n_obs, 'n_val': val.n_obs, 'n_training_controls': int(train.obs.condition.eq('ctrl').sum()),
        'test_expression_used': False,
    })
    print(f'Prepared {work}: {train.n_obs} train / {val.n_obs} validation cells, {len(panel)} genes')


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(args):
    from gears import GEARS, PertData

    work = args.work_dir.resolve()
    manifest = json.loads((work / 'vcc.json').read_text())
    if args.model_dir.exists() and any(args.model_dir.iterdir()):
        raise ValueError('--model-dir must be empty for a new training run')
    seed_all(args.seed)
    data = PertData(str(work.parent), gene_set_path=str(work / 'pert_genes.pkl'))
    data.load(data_path=str(work))
    if data.pert_names.tolist() != manifest['pert_genes'] or data.adata.var.gene_name.tolist() != manifest['genes']:
        raise ValueError('Prepared gene/perturbation order has changed; prepare into a new directory')
    data.prepare_split(split='custom', split_dict_path=str(work / 'split.pkl'), seed=args.seed)
    data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    if not len(data.dataloader['train_loader']):
        raise ValueError('Training split has fewer cells than --batch-size (GEARS uses drop_last=True)')
    # The official test set is evaluated separately, after checkpoint selection.
    data.dataloader.pop('test_loader', None)
    model = GEARS(data, device=args.device)
    model.model_initialize(hidden_size=args.hidden_size)
    model.train(epochs=args.epochs, lr=args.lr)
    args.model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_dir))
    manifest['training_arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.model_dir / 'vcc.json', manifest)
    print(f'Saved best validation checkpoint to {args.model_dir}')


def predict(args):
    import torch
    from torch_geometric.data import Batch, Data
    from gears.model import GEARS_Model

    if args.output.exists():
        raise ValueError('Prediction already exists; choose a new --output path')
    manifest = json.loads((args.model_dir / 'vcc.json').read_text())
    prepared = json.loads((args.work_dir / 'vcc.json').read_text())
    if any(manifest[k] != prepared[k] for k in ['genes', 'pert_genes', 'cell_type', 'arguments']):
        raise ValueError('Checkpoint and prepared VCC data do not match')
    mapping = read_mapping(args.gene_map)
    conditions = read_conditions(args.conditions or args.work_dir / 'conditions.csv',
                                 mapping, manifest['cell_type'])
    check_coverage(conditions.target_gene, manifest['pert_genes'])
    controls = ad.read_h5ad(args.work_dir / 'control.h5ad')
    if not controls.n_obs or controls.var_names.tolist() != manifest['genes']:
        raise ValueError('Training controls do not match checkpoint genes')
    with (args.model_dir / 'config.pkl').open('rb') as handle:
        config = pickle.load(handle)
    config.update(device=args.device, num_genes=len(manifest['genes']), num_perts=len(manifest['pert_genes']))
    seed_all(args.seed)
    model = GEARS_Model(config).to(args.device)
    model.load_state_dict(torch.load(args.model_dir / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    rng = np.random.default_rng(args.seed)
    pert_index = {gene: i for i, gene in enumerate(manifest['pert_genes'])}
    total = int(conditions.n_cells.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Bound RAM use while retaining every requested cell, without predict()'s mean reduction.
    with tempfile.TemporaryDirectory(prefix='.vcc-predict-', dir=args.output.parent) as temporary:
        matrix = np.lib.format.open_memmap(Path(temporary) / 'prediction.npy', mode='w+',
                                          dtype=np.float32, shape=(total, len(manifest['genes'])))
        offset, negative = 0, 0
        with torch.inference_mode():
            for row in conditions.itertuples(index=False):
                for start in range(0, row.n_cells, args.batch_size):
                    size = min(args.batch_size, row.n_cells - start)
                    basal = controls.X[rng.integers(controls.n_obs, size=size)].toarray()
                    graphs = [Data(x=torch.from_numpy(x.reshape(-1, 1)),
                                   pert_idx=[pert_index[row.target_gene]]) for x in basal]
                    result = model(Batch.from_data_list(graphs).to(args.device))
                    if config['uncertainty']:
                        result = result[0]
                    values = result.cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError(f'Nonfinite predictions for {row.target_gene}')
                    negative += int((values < 0).sum())
                    matrix[offset:offset + size] = np.maximum(values, 0)
                    offset += size
                print(f'{row.target_gene}: {row.n_cells} cells', flush=True)
        labels = np.repeat(conditions.target_gene.to_numpy(), conditions.n_cells.to_numpy())
        obs = pd.DataFrame({'target_gene': labels, 'cell_type': manifest['cell_type']},
                           index=[f'gears_{i}' for i in range(total)])
        # AnnData's IO registry accepts ndarray, not the memmap subclass. This is a view.
        output = ad.AnnData(np.asarray(matrix), obs=obs,
                            var=pd.DataFrame({'gene_name': manifest['genes']}, index=manifest['genes']))
        output.uns['evaluation'] = {'expression_scale': 'lognorm', 'producer': 'GEARS VCC',
                                    'negative_values_clipped': negative,
                                    'gene_panel_sha256': manifest['gene_panel_sha256']}
        output.uns['vcc_run'] = {'seed': args.seed, 'model_dir': str(args.model_dir.resolve()),
                                  'work_dir': str(args.work_dir.resolve()), 'control_source': 'training only',
                                  'batch_size': args.batch_size,
                                  'conditions': str((args.conditions or args.work_dir / 'conditions.csv').resolve())}
        temporary_h5ad = Path(temporary) / 'prediction.h5ad'
        output.write_h5ad(temporary_h5ad, compression='gzip')
        temporary_h5ad.replace(args.output)
        del output, matrix
    print(f'Saved {args.output}: {total} cells; expression_scale=lognorm')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'train', 'predict']:
        p = sub.add_parser(command)
        p.add_argument('--work-dir', type=Path, default=REPO / 'data/vcc')
        p.add_argument('--seed', type=int, default=1)
        if command == 'prepare':
            p.add_argument('--data-root', type=Path, default=Path('/data/dataset/lingshucell/vcc'))
            p.add_argument('--train-h5ad', type=Path)
            p.add_argument('--val-h5ad', type=Path)
            p.add_argument('--gene-panel', type=Path, default=LINGSHU / 'data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json')
            p.add_argument('--target-key', default='target_gene')
            p.add_argument('--cell-type', default='H1')
            p.add_argument('--layer', help='Read expression from this layer instead of X')
            p.add_argument('--input-scale', choices=['counts', 'lognorm'], default='counts')
        if command != 'train':
            p.add_argument('--gene-map', type=Path, default=LINGSHU / 'data/vcc/gene_id_name.csv')
            p.add_argument('--conditions', type=Path, help='CSV with cell_type, target_gene, n_cells')
        if command != 'prepare':
            p.add_argument('--model-dir', type=Path, required=True)
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--batch-size', type=int, default=16)
        if command == 'train':
            p.add_argument('--epochs', type=int, default=20)
            p.add_argument('--hidden-size', type=int, default=64)
            p.add_argument('--lr', type=float, default=1e-3)
        if command == 'predict':
            p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ['batch_size', 'epochs', 'hidden_size', 'lr']:
        if hasattr(args, key) and (not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.seed < 0 or args.seed >= 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    if args.command == 'train' and args.batch_size < 2:
        parser.error('--batch-size must be at least 2 for GEARS batch normalization')
    {'prepare': prepare, 'train': train_model, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    main()
