## Imports
from cytograph.species import Species

def rna_barcodes_to_atac(ds):
    barcodes_rna = []
    for barcode in ds.ca.CellID:
        sample = '_'.join(barcode.split('_')[:2])
        b = barcode.split(':')[-1][:-1]
        barcodes_rna.append(':'.join([sample,b]))
    return barcodes_rna

class CellCycleAnnotator:
    def __init__(self, species: Species) -> None:
        self.species = species

    def fit(self, ds: loompy.LoomConnection, layer:str='', recompute_UMIs:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        g1_indices = np.isin(ds.ra.Gene, self.species.genes.g1)
        s_indices = np.isin(ds.ra.Gene, self.species.genes.s)
        g2m_indices = np.isin(ds.ra.Gene, self.species.genes.g2m)
        g1_totals = ds[layer][g1_indices, :].sum(axis=0)
        s_totals = ds[layer][s_indices, :].sum(axis=0)
        g2m_totals = ds[layer][g2m_indices, :].sum(axis=0)
        if recompute_UMIs:
            ds.ca.TotalUMIs = ds[layer].map([np.sum], axis=1)[0]
            total_umis = ds.ca.TotalUMIs
        else:
            if "TotalUMIs" in ds.ca:
                TotalUMIs = ds.ca.TotalUMIs  # From loompy
            else:
                total_umis = ds.ca.TotalUMI  # From cytograph
        return (g1_totals / total_umis, s_totals / total_umis, g2m_totals / total_umis)

    def annotate(self, ds: loompy.LoomConnection, layer:str='', recompute_UMIs=False) -> None:
        """
        Compute the fraction of UMIs that arise from cell cycle genes
        """
        (g1, s, g2m) = self.fit(ds, layer=layer, recompute_UMIs=recompute_UMIs)
        ds.ca.CellCycle_G1 = g1.reshape(-1)
        ds.ca.CellCycle_S = s.reshape(-1)
        ds.ca.CellCycle_G2M = g2m.reshape(-1)
        ds.ca.CellCycle = (g1 + s + g2m).reshape(-1)
        ds.ca.IsCycling = ds.ca.CellCycle > 0.01  # This threshold will depend on the species and on the gene list; 1% is good for the current human and mouse gene sets
