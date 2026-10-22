"""Small AnnData/IO checks; neural network execution is stubbed, not trained."""

from contextlib import nullcontext
import json
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import vcc


@pytest.fixture
def genes():
    return [f'G{i}' for i in range(20)]


def save_source(path, genes, labels, *, values=None, layer=False):
    values = np.asarray(values if values is not None else
                        np.arange(1, len(labels) * len(genes) + 1).reshape(len(labels), len(genes)), dtype=np.float32)
    obs = pd.DataFrame({'target_gene': labels, 'cell_type': 'H1'},
                       index=[f'cell{i}' for i in range(len(labels))])
    source = ad.AnnData(sparse.csr_matrix(values), obs=obs, var=pd.DataFrame(index=genes))
    if layer:
        source.layers['counts'] = source.X.copy()
        source.X = source.X * 0.01
    source.write_h5ad(path)
    return values


def load(path, genes, **kwargs):
    options = dict(split='train', target_key='target_gene', cell_type='H1', layer=None, input_scale='counts')
    options.update(kwargs)
    return vcc.load_adata(path, genes, {}, **options)


def test_reorders_ensembl_genes_and_reads_counts_layer(tmp_path, genes):
    ids = [f'ENSG{i:011d}.2' for i in range(20)]
    path = tmp_path / 'train.h5ad'
    values = save_source(path, ids[::-1] + ['ENSG99999999999'],
                         ['non-targeting', 'ENSG00000000000'], layer=True)
    mapping = {key.removesuffix('.2'): gene for key, gene in zip(ids, genes)}
    result = vcc.load_adata(path, genes, mapping, split='train', target_key='target_gene',
                            cell_type='H1', layer='counts', input_scale='counts')
    expected = values[:, :20][:, ::-1]
    expected = np.log1p(expected / expected.sum(axis=1, keepdims=True) * 10000)
    np.testing.assert_allclose(result.X.toarray(), expected, rtol=1e-6)
    assert result.var_names.tolist() == genes
    assert result.obs.condition.tolist() == ['ctrl', 'G0+ctrl']
    assert sparse.isspmatrix_csr(result.X)


def test_validation_controls_are_excluded_and_lognorm_not_transformed(tmp_path, genes):
    path = tmp_path / 'val.h5ad'
    values = np.full((2, 20), 1.25, dtype=np.float32)
    save_source(path, genes, ['non-targeting', 'G1'], values=values)
    with pytest.raises(ValueError, match='noninteger counts'):
        load(path, genes)
    result = load(path, genes, split='val', input_scale='lognorm')
    assert result.obs.condition.tolist() == ['G1+ctrl']
    np.testing.assert_array_equal(result.X.toarray(), values[1:])


@pytest.mark.parametrize('train,val,test', [
    (['ctrl', 'G0+ctrl'], ['G0+ctrl'], ['G2']),
    (['ctrl', 'G0+ctrl'], ['G1+ctrl'], ['G0']),
    (['ctrl', 'G0+ctrl'], ['G1+ctrl'], ['G1']),
])
def test_rejects_perturbation_leakage(train, val, test):
    with pytest.raises(ValueError, match='disjoint'):
        vcc.fixed_split(train, val, test)


def test_missing_panel_and_go_targets_fail_explicitly(tmp_path, genes):
    path = tmp_path / 'train.h5ad'
    save_source(path, genes[:-1], ['ctrl', 'G0'])
    with pytest.raises(ValueError, match='missing panel genes=.*G19'):
        load(path, genes)
    with pytest.raises(ValueError, match='GO graph.*G19'):
        vcc.check_coverage(['G0', 'G19'], ['G0'])


def test_conditions_mapping_and_invalid_cell_budgets(tmp_path):
    path = tmp_path / 'conditions.csv'
    pd.DataFrame({'target_gene': ['non-targeting', 'ENSG001', 'G2+ctrl'],
                  'n_cells': [8, 5, 3]}).to_csv(path, index=False)
    table = vcc.read_conditions(path, {'ENSG001': 'G1'}, 'H1')
    assert table.target_gene.tolist() == ['G1', 'G2']
    assert table.n_cells.tolist() == [5, 3]
    pd.DataFrame({'target_gene': ['G1'], 'n_cells': [1.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='positive integer'):
        vcc.read_conditions(path, {}, 'H1')


def test_prepare_preserves_official_split_without_test_expression(tmp_path, genes, monkeypatch):
    train, val = tmp_path / 'train.h5ad', tmp_path / 'val.h5ad'
    save_source(train, genes, ['ctrl', 'G0', 'G0'])
    save_source(val, genes, ['ctrl', 'G1', 'G1'])
    conditions, panel, mapping = [tmp_path / name for name in ['conditions.csv', 'panel.txt', 'map.tsv']]
    pd.DataFrame({'target_gene': ['G2'], 'n_cells': [5]}).to_csv(conditions, index=False)
    panel.write_text('\n'.join(genes))
    mapping.write_text('ENSG001\tG0\n')

    class PertData:
        def __init__(self, root, gene_set_path):
            self.root = Path(root)
            self.pert_names = np.array(pickle.loads(Path(gene_set_path).read_bytes()))
            self.gene2go = dict.fromkeys(self.pert_names, {'GO:1'})

        def new_data_process(self, dataset_name, adata):
            assert set(adata.obs.condition) == {'ctrl', 'G0+ctrl', 'G1+ctrl'}
            assert adata.obs.condition.eq('ctrl').sum() == 1
            adata.write_h5ad(self.root / dataset_name / 'perturb_processed.h5ad')

    monkeypatch.setitem(sys.modules, 'gears', SimpleNamespace(PertData=PertData))
    work = tmp_path / 'vcc'
    vcc.main(['prepare', '--work-dir', str(work), '--train-h5ad', str(train),
              '--val-h5ad', str(val), '--conditions', str(conditions),
              '--gene-panel', str(panel), '--gene-map', str(mapping)])
    assert not (tmp_path / 'test.h5ad').exists()
    splits = pickle.loads((work / 'split.pkl').read_bytes())
    assert splits == {'train': ['ctrl', 'G0+ctrl'], 'val': ['G1+ctrl'], 'test': []}
    manifest = json.loads((work / 'vcc.json').read_text())
    assert manifest['test_expression_used'] is False
    assert manifest['n_train'] == 3 and manifest['n_val'] == 2
    assert manifest['pert_genes'] == ['G0', 'G1', 'G2']
    controls = ad.read_h5ad(work / 'control.h5ad')
    assert controls.n_obs == 1 and controls.obs.vcc_split.tolist() == ['train']


def test_per_cell_export_keeps_all_batches_and_declares_scale(tmp_path, genes, monkeypatch):
    work, checkpoint = tmp_path / 'vcc', tmp_path / 'model'
    work.mkdir()
    checkpoint.mkdir()
    manifest = dict(genes=genes, pert_genes=['G0', 'G1'], cell_type='H1',
                    arguments={'input_scale': 'counts'}, gene_panel_sha256='panel-hash')
    for directory in [work, checkpoint]:
        vcc.write_json(directory / 'vcc.json', manifest)
    with (checkpoint / 'config.pkl').open('wb') as handle:
        pickle.dump({'uncertainty': False}, handle)
    mapping = tmp_path / 'map.tsv'
    mapping.write_text('ENSG001\tG0\n')
    ad.AnnData(sparse.csr_matrix(np.array([[0] + [2] * 19, [0] + [3] * 19], dtype=np.float32)),
               var=pd.DataFrame(index=genes)).write_h5ad(work / 'control.h5ad')
    pd.DataFrame({'target_gene': ['G0', 'G1'], 'n_cells': [5, 3]}).to_csv(work / 'conditions.csv', index=False)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Batch:
        @classmethod
        def from_data_list(cls, graphs):
            batch = cls()
            batch.graphs = graphs
            return batch

        def to(self, device):
            return self

    class Model:
        def __init__(self, config):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, weights):
            pass

        def eval(self):
            pass

        def __call__(self, batch):
            return Tensor(np.stack([graph.x.ravel() + graph.pert_idx[0] - 1 for graph in batch.graphs]))

    fake_torch = SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False),
                                 inference_mode=nullcontext, from_numpy=lambda array: array, load=lambda *a, **kw: {})
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch_geometric', ModuleType('torch_geometric'))
    monkeypatch.setitem(sys.modules, 'torch_geometric.data', SimpleNamespace(Batch=Batch, Data=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'gears', ModuleType('gears'))
    monkeypatch.setitem(sys.modules, 'gears.model', SimpleNamespace(GEARS_Model=Model))
    output = tmp_path / 'prediction.h5ad'
    command = ['predict', '--work-dir', str(work), '--model-dir', str(checkpoint),
               '--gene-map', str(mapping), '--device', 'cpu', '--batch-size', '2', '--output', str(output)]
    vcc.main(command)
    result = ad.read_h5ad(output)
    assert result.shape == (8, 20)
    assert result.obs.target_gene.tolist() == ['G0'] * 5 + ['G1'] * 3
    assert result.var_names.tolist() == genes
    assert result.uns['evaluation']['expression_scale'] == 'lognorm'
    assert result.uns['evaluation']['negative_values_clipped'] == 5
    assert (result.X >= 0).all()
    first = result.X.copy()
    output.unlink()
    vcc.main(command)
    np.testing.assert_array_equal(ad.read_h5ad(output).X, first)
    assert not list(tmp_path.glob('.vcc-predict-*'))
