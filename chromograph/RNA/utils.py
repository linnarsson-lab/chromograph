## Imports

def rna_barcodes_to_atac(ds):
    barcodes_rna = []
    for barcode in ds.ca.CellID:
        sample = '_'.join(barcode.split('_')[:2])
        b = barcode.split(':')[-1][:-1]
        barcodes_rna.append(':'.join([sample,b]))
    return barcodes_rna
