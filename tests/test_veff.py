import pytest
from kipoi_enformer.benchmark import VeffBenchmark
from pathlib import Path
import test_enformer


@pytest.fixture
def variant_effects_paths(chr22_example_files, output_dir: Path,
                          enformer_tracks_path: Path, gtex_tissue_mapper_path: Path):
    return {
        'enformer_logsumexp': test_enformer.test_calculate_veff(chr22_example_files, output_dir, enformer_tracks_path,
                                                                gtex_tissue_mapper_path,
                                                                aggregation_mode='logsumexp',
                                                                upstream_tss=100, downstream_tss=500, size=10),
        'enformer_weighted_sum': test_enformer.test_calculate_veff(chr22_example_files, output_dir,
                                                                   enformer_tracks_path,
                                                                   gtex_tissue_mapper_path,
                                                                   aggregation_mode='weighted_sum',
                                                                   upstream_tss=100, downstream_tss=500, size=10)
    }


@pytest.fixture
def benchmark(chr22_example_files):
    return VeffBenchmark(annotation_path=chr22_example_files['gtex_annotation'],
                         genotypes_path=chr22_example_files['gtex_variants'])


@pytest.mark.parametrize("predictor", [
    'enformer_logsumexp', 'enformer_weighted_sum'
])
def test_benchmark(chr22_example_files, benchmark, predictor, variant_effects_paths, output_dir: Path):
    gene_variant_effects_path = variant_effects_paths[predictor]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{predictor}_benchmark.parquet'
    benchmark.run([gene_variant_effects_path], output_path=output_path)

    assert output_path.exists()
