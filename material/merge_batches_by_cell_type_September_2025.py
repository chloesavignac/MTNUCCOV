import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import warnings

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- Load cell annotations ----------------------
annot = pd.read_csv('cell-annotation.csv')

# Get unique cell type categories
cells_cat = list(annot['cell.type'].unique())

# ---------------------- Load and clean batch list ----------------------
with open('batch_list.txt') as f:
    batch_list = f.readlines()

# Remove trailing slashes/newlines and store cleaned batch names
batches = [b.strip('/\n') for b in batch_list]

# ---------------------- Function to merge batches by cell type ----------------------
def merge_batches(cell_type, path):
    """
    Merge gene expression data across batches for a specific cell type.

    Args:
        cell_type (str): Cell type to subset.
        path (str): Path to directory containing batch files.

    Returns:
        pd.DataFrame: Merged dataframe with cells of the specified type across all batches.
    """
    print(f'Processing cell type: {cell_type}')
    merged_df = pd.DataFrame()

    for batch in tqdm(batches, desc='Merging batches'):
        # Load batch-specific AnnData object
        adata = sc.read_h5ad(f'{path}/seurat/{batch}_seurat_qc.h5ad')
        
        # Columns are the top 2000 genes for this batch
        cols = list(adata.var.features)

        # Convert AnnData matrix to DataFrame
        if isinstance(adata.X, np.ndarray):
            cells_df = pd.DataFrame(data=adata.X, columns=cols)
        else:
            # Convert sparse matrix to dense
            cells_df = pd.DataFrame(data=adata.X.toarray(), columns=cols)

        cells_df.reset_index(drop=True, inplace=True)

        # Add barcodes for cell identification
        obs_df = adata.obs.reset_index()
        obs_df['barcode'] = batch + '_' + obs_df['index']
        cells_df['barcode'] = obs_df['barcode']

        # Subset cells of the specified cell type
        cell_type_annot = annot[annot['cell.type'] == cell_type]
        cells_df = cell_type_annot.merge(cells_df, on='barcode', how='inner')

        # Skip empty batches
        if cells_df.empty:
            continue

        # Append to merged dataframe
        merged_df = pd.concat([merged_df, cells_df], axis=0, ignore_index=True, join='outer')

    return merged_df

# ---------------------- Merge all cell types and save ----------------------
path = 'ROSMAP_seurat_QC'

for cell_type in cells_cat:
    # Merge all batches for this cell type
    cells_df = merge_batches(cell_type, path)

    # Save metadata (first 5 columns)
    meta_data = cells_df.iloc[:, :5]
    meta_data.to_csv(f'{path}/by_cell_type/with_nan/{cell_type}_meta_data.csv', index=False)

    # Save gene expression array (columns from 5 onward)
    genes_arr = np.array(cells_df.iloc[:, 5:])
    np.save(f'{path}/by_cell_type/with_nan/{cell_type}_genes_arr.npy', genes_arr)

    # Save gene names
    genes_list = np.array(cells_df.iloc[:, 5:].columns)
    np.save(f'{path}/by_cell_type/with_nan/{cell_type}_genes_names.npy', genes_list)

print('Done!')
